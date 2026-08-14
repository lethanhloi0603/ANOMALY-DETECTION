"""Train a TCN-Transformer Autoencoder from versioned model-ready windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from cli.score import (
    FEATURE_SCHEMA_VERSION,
    FEATURE_VALUE_SPACE,
    SEQUENCE_SCHEMA_VERSION,
    SUPPORTED_FRAMEWORK_VERSIONS,
    _load_framework_config,
    _validate_complete_source_attestation,
    _validate_prepared_reference_source,
    _validate_store_reference_source,
)
from insider_ml.artifacts import atomic_write_json
from insider_ml.contracts import TRAIN_START
from insider_ml.dataset import (
    SQLiteWindowDataset,
    WindowDataset,
    load_prepared_windows,
)
from insider_ml.model import build_model, load_model_config
from insider_ml.training import train_epoch


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Prepared NPZ training windows")
    source.add_argument("--store", type=Path, help="Disk-backed daily tensor SQLite store")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt checkpoint")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--framework-config",
        type=Path,
        default=root / "backend" / "config" / "framework.v6.json",
    )
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


def _weekly_train_indices(windows: Any) -> list[int]:
    indices = [
        index
        for index, (user_id, raw_day) in enumerate(
            zip(
                windows.sample_user_ids,
                windows.sample_end_days,
                strict=True,
            )
        )
        if (date.fromisoformat(str(raw_day)) - TRAIN_START).days % 7
        == int.from_bytes(
            hashlib.sha256(str(user_id).encode()).digest()[:4],
            "little",
        )
        % 7
    ]
    if not indices:
        raise ValueError("prepared Train NPZ has no WEEKLY_TRAIN endpoints")
    return indices


def _compose_training_source_attestation(
    *,
    source_kind: str,
    source: str,
    source_sha256: str,
    validated_attestation: Mapping[str, Any] | None = None,
    incomplete_reason_code: str | None = None,
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    """Normalize strict source evidence or an explicit non-production reason."""

    supported_kinds = {"prepared_windows_npz", "sqlite_user_day_store"}
    if source_kind not in supported_kinds:
        raise ValueError(f"unsupported training source kind {source_kind!r}")
    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256
    ):
        raise ValueError("training source SHA-256 must contain 64 hexadecimal characters")

    if validated_attestation is not None:
        if incomplete_reason_code is not None or incomplete_reason is not None:
            raise ValueError("complete training attestation cannot include an incomplete reason")
        if (
            validated_attestation.get("schema_version")
            != "reference-source-attestation.v1"
            or validated_attestation.get("kind") != source_kind
            or validated_attestation.get("complete_train") is not True
        ):
            raise ValueError("strict validator returned an invalid Train attestation")
        validated_source = validated_attestation.get("source")
        if validated_source is not None and validated_source != source:
            raise ValueError("strict validator returned a different training source")
        validated_sha256 = validated_attestation.get("source_sha256")
        if validated_sha256 is not None and validated_sha256 != source_sha256:
            raise ValueError("training source changed while it was being attested")
        result = dict(validated_attestation)
        result["source"] = source
        result["source_sha256"] = source_sha256
        _validate_complete_source_attestation(
            result,
            require_source_sha256=True,
        )
        return result

    if not incomplete_reason_code or not incomplete_reason:
        raise ValueError("incomplete training attestation requires a reason code and reason")
    return {
        "schema_version": "reference-source-attestation.v1",
        "kind": source_kind,
        "complete_train": False,
        "source": source,
        "source_sha256": source_sha256,
        "reason_code": incomplete_reason_code,
        "reason": incomplete_reason,
    }


def _build_training_source_attestation(
    source_path: Path,
    *,
    source_kind: str,
    max_samples: int | None,
    preprocessing_checksum: str,
    scaler_checksum: str,
    sample_count: int,
    feature_schema_version: str,
    sequence_schema_version: str,
) -> dict[str, Any]:
    """Attest complete Train coverage without blocking smoke/research training."""

    resolved_source = str(source_path.resolve())
    source_sha256 = _sha256(source_path)
    if source_kind == "sqlite_user_day_store" and max_samples is not None:
        return _compose_training_source_attestation(
            source_kind=source_kind,
            source=resolved_source,
            source_sha256=source_sha256,
            incomplete_reason_code="TRAINING_SAMPLE_CAP_APPLIED",
            incomplete_reason=(
                f"--max-samples={max_samples} limits the Train endpoint set; "
                "this checkpoint is smoke/research-only"
            ),
        )

    try:
        if source_kind == "sqlite_user_day_store":
            validated = _validate_store_reference_source(source_path)
        elif source_kind == "prepared_windows_npz":
            validated = _validate_prepared_reference_source(
                source_path,
                preprocessing_checksum=preprocessing_checksum,
                scaler_checksum=scaler_checksum,
                sample_count=sample_count,
                feature_schema_version=feature_schema_version,
                sequence_schema_version=sequence_schema_version,
            )
        else:
            raise ValueError(f"unsupported training source kind {source_kind!r}")
    except ValueError as exc:
        return _compose_training_source_attestation(
            source_kind=source_kind,
            source=resolved_source,
            source_sha256=source_sha256,
            incomplete_reason_code="SOURCE_NOT_COMPLETE_TRAIN",
            incomplete_reason=str(exc),
        )

    return _compose_training_source_attestation(
        source_kind=source_kind,
        source=resolved_source,
        source_sha256=source_sha256,
        validated_attestation=validated,
    )


def _training_contract_fields(
    training_source_attestation: Mapping[str, Any],
    *,
    framework_config: Mapping[str, Any],
    framework_config_checksum: str,
    training_endpoint_policy: str,
) -> dict[str, Any]:
    """Return the immutable Train-only fields shared by checkpoint and manifest."""

    if not isinstance(training_source_attestation.get("complete_train"), bool):
        raise ValueError("training source attestation requires complete_train boolean")
    framework_version = framework_config.get("schema_version")
    if (
        framework_version not in SUPPORTED_FRAMEWORK_VERSIONS
        or len(framework_config_checksum) != 64
        or any(
            character not in "0123456789abcdef"
            for character in framework_config_checksum
        )
        or training_endpoint_policy not in {"WEEKLY_TRAIN", "ALL"}
    ):
        raise ValueError("training contract does not match a supported framework")
    return {
        "fit_split": "TRAIN",
        "frozen_after_train": True,
        "framework_schema_version": framework_version,
        "framework_config_sha256": framework_config_checksum,
        "training_endpoint_policy": training_endpoint_policy,
        "training_source_attestation": dict(training_source_attestation),
    }


def _validate_resume_training_contract(
    prior: Mapping[str, Any],
    current_training_contract: Mapping[str, Any],
) -> None:
    for field_name, expected in current_training_contract.items():
        if prior.get(field_name) != expected:
            raise SystemExit(f"resume checkpoint {field_name} does not match current source")


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise SystemExit("epochs, batch-size, and learning-rate must be positive")
    if args.input and args.max_samples is not None:
        raise SystemExit("--max-samples is only supported with --store")

    torch.manual_seed(args.seed)
    framework_config = _load_framework_config(args.framework_config)
    framework_config_checksum = _sha256(args.framework_config)
    training_sampling = framework_config.get("training_sampling")
    if (
        not isinstance(training_sampling, dict)
        or str(training_sampling.get("endpoint_policy", "")).upper()
        != "WEEKLY_TRAIN"
        or training_sampling.get("reference_fit_uses_all_train_endpoints") is not True
    ):
        raise SystemExit("training requires the locked framework sampling contract")
    expected_training_endpoint_policy = "WEEKLY_TRAIN"
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
        source_sample_count = int(windows.feature_values.shape[0])
        dataset = Subset(
            WindowDataset(windows),
            _weekly_train_indices(windows),
        )
        training_endpoint_policy = expected_training_endpoint_policy
        preprocessing_checksum = str(windows.preprocessing_checksum.item())
        scaler_checksum = str(windows.scaler_checksum.item())
        feature_schema_version = str(windows.feature_schema_version.item())
        sequence_schema_version = str(windows.sequence_schema_version.item())
        feature_value_space = str(windows.feature_value_space.item())
        source_path = args.input
    else:
        training_endpoint_policy = args.endpoint_policy.upper()
        dataset = SQLiteWindowDataset(
            args.store,
            split="TRAIN",
            max_samples=args.max_samples,
            endpoint_policy=args.endpoint_policy,
        )
        preprocessing_checksum = str(dataset.preprocessing_checksum)
        scaler_checksum = str(dataset.scaler_checksum)
        source_sample_count = len(dataset)
        feature_schema_version = FEATURE_SCHEMA_VERSION
        sequence_schema_version = SEQUENCE_SCHEMA_VERSION
        feature_value_space = FEATURE_VALUE_SPACE
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
    source_kind = "prepared_windows_npz" if args.input else "sqlite_user_day_store"
    training_source_attestation = _build_training_source_attestation(
        source_path,
        source_kind=source_kind,
        max_samples=args.max_samples,
        preprocessing_checksum=preprocessing_checksum,
        scaler_checksum=scaler_checksum,
        sample_count=source_sample_count,
        feature_schema_version=feature_schema_version,
        sequence_schema_version=sequence_schema_version,
    )
    training_contract = _training_contract_fields(
        training_source_attestation,
        framework_config=framework_config,
        framework_config_checksum=framework_config_checksum,
        training_endpoint_policy=training_endpoint_policy,
    )

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
        if prior.get("training_endpoint_policy") != training_endpoint_policy:
            raise SystemExit("resume checkpoint endpoint policy does not match")
        _validate_resume_training_contract(prior, training_contract)
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
            **training_contract,
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
        **training_contract,
        "created_at": datetime.now(UTC).isoformat(),
        "input": str(source_path.resolve()),
        "input_kind": "npz" if args.input else "sqlite_user_day_store",
        "input_sha256": training_source_attestation["source_sha256"],
        "feature_schema_version": feature_schema_version,
        "sequence_schema_version": sequence_schema_version,
        "preprocessing_checksum": preprocessing_checksum,
        "scaler_checksum": scaler_checksum,
        "feature_value_space": feature_value_space,
        "training_samples": len(dataset),
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
