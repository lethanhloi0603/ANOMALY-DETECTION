"""Train a TCN-Transformer Autoencoder from versioned model-ready windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from insider_ml.artifacts import atomic_write_json
from insider_ml.dataset import (
    SQLiteWindowDataset,
    WindowDataset,
    load_prepared_windows,
)
from insider_ml.model import build_model, load_model_config
from insider_ml.training import train_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Prepared NPZ training windows")
    source.add_argument("--store", type=Path, help="Disk-backed daily tensor SQLite store")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt checkpoint")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Smoke-only cap for --store; omit for the complete Train split",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted checkpoint and save after every epoch",
    )
    parser.add_argument(
        "--endpoint-policy",
        choices=("weekly_train", "all"),
        default="weekly_train",
        help=(
            "Store-only Train endpoint sampling. weekly_train is label-free, "
            "user-staggered and still reconstructs all 30 days in each window."
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise SystemExit("epochs, batch-size, and learning-rate must be positive")
    if args.input and args.max_samples is not None:
        raise SystemExit("--max-samples is only supported with --store")

    torch.manual_seed(args.seed)
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    device = torch.device(device_name)
    config = load_model_config(args.config)
    if args.input:
        windows = load_prepared_windows(args.input)
        observed_splits = {str(value).upper() for value in windows.sample_splits}
        if observed_splits != {"TRAIN"}:
            raise SystemExit(
                "training NPZ must contain TRAIN samples only; "
                f"observed splits={sorted(observed_splits)}"
            )
        dataset = WindowDataset(windows)
        preprocessing_checksum = str(windows.preprocessing_checksum.item())
        scaler_checksum = str(windows.scaler_checksum.item())
        feature_schema_version = str(windows.feature_schema_version.item())
        sequence_schema_version = str(windows.sequence_schema_version.item())
        feature_value_space = str(windows.feature_value_space.item())
        source_path = args.input
    else:
        dataset = SQLiteWindowDataset(
            args.store,
            split="TRAIN",
            max_samples=args.max_samples,
            endpoint_policy=args.endpoint_policy,
        )
        preprocessing_checksum = str(dataset.preprocessing_checksum)
        scaler_checksum = str(dataset.scaler_checksum)
        feature_schema_version = "feature128.v5"
        sequence_schema_version = "sequence7.v4"
        feature_value_space = "robust_scaled"
        source_path = args.store
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.input is not None,
        num_workers=0,
    )
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_config = config["loss"]
    component_weights = loss_config.get("sequence_components", {})
    if (
        loss_config.get("sequence_component_reduction")
        != "equal_mean_over_available_components"
        or not component_weights
        or any(float(value) != 1.0 for value in component_weights.values())
    ):
        raise SystemExit(
            "training implementation currently requires equal unit sequence components"
        )
    endpoint_policy = "NPZ_AS_PREPARED" if args.input else dataset.endpoint_policy

    args.output.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    if args.resume and args.output.is_file():
        prior = torch.load(args.output, map_location=device, weights_only=True)
        if prior.get("schema_version") != config["schema_version"]:
            raise SystemExit("resume checkpoint model schema does not match")
        if prior.get("config") != config:
            raise SystemExit("resume checkpoint model config does not match")
        if prior.get("preprocessing_checksum") != preprocessing_checksum:
            raise SystemExit("resume checkpoint preprocessing checksum does not match")
        if prior.get("scaler_checksum") != scaler_checksum:
            raise SystemExit("resume checkpoint scaler checksum does not match")
        if prior.get("training_samples") != len(dataset):
            raise SystemExit("resume checkpoint training sample count does not match")
        if prior.get("training_endpoint_policy") != endpoint_policy:
            raise SystemExit("resume checkpoint endpoint policy does not match")
        if float(prior.get("learning_rate", -1)) != args.learning_rate:
            raise SystemExit("resume checkpoint learning rate does not match")
        model.load_state_dict(prior["model_state_dict"])
        if "optimizer_state_dict" not in prior:
            raise SystemExit("resume checkpoint has no optimizer state")
        optimizer.load_state_dict(prior["optimizer_state_dict"])
        history = list(prior.get("training_history", []))
        if "torch_rng_state" in prior:
            torch.set_rng_state(prior["torch_rng_state"].cpu())
    start_epoch = len(history) + 1
    for epoch in range(start_epoch, args.epochs + 1):
        metrics = train_epoch(
            model,
            loader,
            optimizer,
            device=device,
            feature_weight=float(loss_config["feature_weight"]),
            sequence_weight=float(loss_config["sequence_weight"]),
        )
        history.append({"epoch": epoch, **metrics})
        print(json.dumps(history[-1], sort_keys=True))
        checkpoint_payload = {
            "schema_version": config["schema_version"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "training_history": history,
            "feature_schema_version": feature_schema_version,
            "sequence_schema_version": sequence_schema_version,
            "preprocessing_checksum": preprocessing_checksum,
            "scaler_checksum": scaler_checksum,
            "feature_value_space": feature_value_space,
            "torch_rng_state": torch.get_rng_state(),
            "training_samples": len(dataset),
            "training_endpoint_policy": endpoint_policy,
            "learning_rate": args.learning_rate,
        }
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        torch.save(checkpoint_payload, temporary)
        temporary.replace(args.output)
    if not history:
        raise SystemExit("training produced no history")
    if not args.output.is_file():
        raise SystemExit("checkpoint was not created")
    manifest = {
        "schema_version": "model-manifest.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "input": str(source_path.resolve()),
        "input_kind": "npz" if args.input else "sqlite_user_day_store",
        "input_sha256": _sha256(source_path),
        "feature_schema_version": feature_schema_version,
        "sequence_schema_version": sequence_schema_version,
        "preprocessing_checksum": preprocessing_checksum,
        "scaler_checksum": scaler_checksum,
        "feature_value_space": feature_value_space,
        "training_samples": len(dataset),
        "training_endpoint_policy": endpoint_policy,
        "checkpoint": str(args.output.resolve()),
        "checkpoint_sha256": _sha256(args.output),
        "device": device_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "final_metrics": history[-1],
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    atomic_write_json(manifest_path, manifest)
    if isinstance(dataset, SQLiteWindowDataset):
        dataset.close()


if __name__ == "__main__":
    main()
