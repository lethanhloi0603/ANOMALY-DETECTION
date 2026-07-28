"""Run last-day inference and frozen Person -> Role -> Global calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from datetime import date
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from insider_ml.cert_data import LdapDirectory
from insider_ml.dataset import (
    SQLiteWindowDataset,
    WindowDataset,
    load_prepared_windows,
)
from insider_ml.inference import (
    ReferenceStats,
    calibrated_tail_score,
    daily_branch_reconstruction_scores,
    fit_reference,
)
from insider_ml.model import build_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--store", type=Path)
    parser.add_argument(
        "--split",
        choices=("TRAIN", "VALIDATION", "TEST"),
        help="Required with --store",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=root / "data" / "raw" / "cert4.2")
    parser.add_argument("--reference-in", type=Path)
    parser.add_argument("--fit-reference-out", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Smoke-only cap for --store; omit for the complete split",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _role(directory: LdapDirectory, user_id: str, day: date) -> str:
    profile = directory.profile(user_id, day)
    return profile.role if profile is not None else "UNKNOWN_ROLE"


def _role_epoch(directory: LdapDirectory, user_id: str, day: date) -> str:
    profile = directory.profile(user_id, day)
    role_start = directory.role_start(user_id, day)
    if profile is None or role_start is None:
        return f"{user_id}|UNKNOWN_ROLE|UNKNOWN"
    return f"{user_id}|{profile.role}|{role_start.isoformat()}"


def _score_rows(
    *,
    windows: Any,
    checkpoint: dict[str, Any],
    batch_size: int,
    device: torch.device,
    directory: LdapDirectory,
) -> list[dict[str, Any]]:
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(WindowDataset(windows), batch_size=batch_size, shuffle=False)
    rows: list[dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = {name: value.to(device) for name, value in raw_batch.items()}
            output = model(**batch)
            scores = daily_branch_reconstruction_scores(output, batch)
            feature = scores.feature[:, -1].cpu().numpy()
            sequence = scores.sequence[:, -1].cpu().numpy()
            sequence_lengths = batch["token_mask"][:, -1].sum(dim=-1).cpu().numpy()
            for local_index in range(feature.shape[0]):
                sample_index = offset + local_index
                user_id = str(windows.sample_user_ids[sample_index])
                day = date.fromisoformat(str(windows.sample_end_days[sample_index]))
                rows.append(
                    {
                        "user_id": user_id,
                        "day": day.isoformat(),
                        "split": str(windows.sample_splits[sample_index]).upper(),
                        "role": _role(directory, user_id, day),
                        "role_epoch": _role_epoch(directory, user_id, day),
                        "feature_raw": (
                            float(feature[local_index])
                            if math.isfinite(float(feature[local_index]))
                            else None
                        ),
                        "sequence_raw": (
                            float(sequence[local_index])
                            if math.isfinite(float(sequence[local_index]))
                            else None
                        ),
                        "sequence_length": int(sequence_lengths[local_index]),
                        "active": bool(sequence_lengths[local_index]),
                        "feature_coverage": float(
                            batch["feature_mask"][local_index, -1].float().mean().cpu()
                        ),
                        "_feature_mask": np.packbits(
                            batch["feature_mask"][local_index, -1]
                            .cpu()
                            .numpy()
                            .astype(np.uint8),
                            bitorder="little",
                        ).tobytes(),
                    }
                )
            offset += feature.shape[0]
    return rows


def _score_store_rows(
    *,
    dataset: SQLiteWindowDataset,
    checkpoint: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows: list[dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = {name: value.to(device) for name, value in raw_batch.items()}
            output = model(**batch)
            scores = daily_branch_reconstruction_scores(output, batch)
            feature = scores.feature[:, -1].cpu().numpy()
            sequence = scores.sequence[:, -1].cpu().numpy()
            sequence_lengths = batch["token_mask"][:, -1].sum(dim=-1).cpu().numpy()
            feature_coverage = (
                batch["feature_mask"][:, -1].float().mean(dim=-1).cpu().numpy()
            )
            for local_index in range(feature.shape[0]):
                sample = dataset.sample_metadata(offset + local_index)
                rows.append(
                    {
                        "user_id": sample.user_id,
                        "day": sample.day,
                        "split": sample.split,
                        "role": sample.role,
                        "role_epoch": sample.role_epoch,
                        "feature_raw": (
                            float(feature[local_index])
                            if math.isfinite(float(feature[local_index]))
                            else None
                        ),
                        "sequence_raw": (
                            float(sequence[local_index])
                            if math.isfinite(float(sequence[local_index]))
                            else None
                        ),
                        "sequence_length": int(sequence_lengths[local_index]),
                        "active": sample.event_count > 0,
                        "feature_coverage": float(feature_coverage[local_index]),
                        "_feature_mask": np.packbits(
                            batch["feature_mask"][local_index, -1]
                            .cpu()
                            .numpy()
                            .astype(np.uint8),
                            bitorder="little",
                        ).tobytes(),
                    }
                )
            offset += feature.shape[0]
    return rows


def _reference_to_json(reference: ReferenceStats) -> dict[str, Any]:
    value = asdict(reference)
    value["sorted_scores"] = list(reference.sorted_scores)
    return value


def _fit_reference_artifact(
    rows: list[dict[str, Any]],
    *,
    checkpoint_path: Path,
    preprocessing_checksum: str,
    scaler_checksum: str,
) -> dict[str, Any]:
    if {row["split"] for row in rows} != {"TRAIN"}:
        raise ValueError("reference fitting accepts Train rows only")
    artifact: dict[str, Any] = {
        "schema_version": "hierarchical-reference.v1",
        "fit_split": "TRAIN",
        "frozen_after_train": True,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "preprocessing_checksum": preprocessing_checksum,
        "scaler_checksum": scaler_checksum,
        "branches": {},
    }
    for branch, score_name in (
        ("feature", "feature_raw"),
        ("sequence", "sequence_raw"),
    ):
        levels: dict[str, dict[str, Any]] = {"PERSON": {}, "ROLE": {}, "GLOBAL": {}}
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
            "PERSON": defaultdict(list),
            "ROLE": defaultdict(list),
            "GLOBAL": defaultdict(list),
        }
        for row in rows:
            if row[score_name] is None:
                continue
            grouped["PERSON"][str(row["role_epoch"])].append(row)
            grouped["ROLE"][str(row["role"])].append(row)
            grouped["GLOBAL"]["GLOBAL"].append(row)
        for level, scopes in grouped.items():
            for scope_key, scope_rows in scopes.items():
                scores = np.asarray([row[score_name] for row in scope_rows], dtype=np.float64)
                stats = fit_reference(level, scope_key, scores)
                days = sorted({str(row["day"]) for row in scope_rows})
                users = sorted({str(row["user_id"]) for row in scope_rows})
                latest_day = date.fromisoformat(days[-1])
                recent_rows = [
                    row
                    for row in scope_rows
                    if 0
                    <= (latest_day - date.fromisoformat(str(row["day"]))).days
                    <= 29
                ]
                feature_support = np.zeros(128, dtype=np.int64)
                for row in scope_rows:
                    packed = np.frombuffer(row["_feature_mask"], dtype=np.uint8)
                    feature_support += np.unpackbits(
                        packed,
                        bitorder="little",
                    )[:128]
                levels[level][scope_key] = {
                    "stats": _reference_to_json(stats),
                    "support": {
                        "observations": len(scope_rows),
                        "users": len(users),
                        "user_ids": users,
                        "span_days": (
                            (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days + 1
                            if days
                            else 0
                        ),
                        "transitions": sum(
                            max(int(row["sequence_length"]) - 1, 0) for row in scope_rows
                        ),
                        "active_days": sum(bool(row["active"]) for row in scope_rows),
                        "mean_feature_coverage": fmean(
                            float(row["feature_coverage"]) for row in scope_rows
                        ),
                        "feature_coverage_ratio_at_20": float(
                            np.mean(feature_support >= 20)
                        ),
                        "feature_coverage_ratio_at_200": float(
                            np.mean(feature_support >= 200)
                        ),
                        "minimum_nonzero_feature_support": int(
                            feature_support[feature_support > 0].min()
                            if np.any(feature_support > 0)
                            else 0
                        ),
                        "last_active_day": max(
                            (
                                str(row["day"])
                                for row in scope_rows
                                if bool(row["active"])
                            ),
                            default=None,
                        ),
                        "recent_30d_observations": len(recent_rows),
                        "recent_30d_transitions": sum(
                            max(int(row["sequence_length"]) - 1, 0)
                            for row in recent_rows
                        ),
                    },
                }
        artifact["branches"][branch] = levels
    return artifact


def _stats(payload: dict[str, Any]) -> ReferenceStats:
    raw = payload["stats"]
    return ReferenceStats(
        level=str(raw["level"]),
        scope_key=str(raw["scope_key"]),
        location=float(raw["location"]),
        scale=float(raw["scale"]),
        observation_count=int(raw["observation_count"]),
        sorted_scores=tuple(float(value) for value in raw["sorted_scores"]),
    )


def _ready(
    *,
    branch: str,
    level: str,
    entry: dict[str, Any],
    user_id: str,
    sequence_length: int,
    day: str,
    role: str,
) -> bool:
    support = entry["support"]
    if branch == "sequence" and sequence_length < 2:
        return False
    if level == "PERSON":
        last_active = support.get("last_active_day")
        stale_gap = (
            (date.fromisoformat(day) - date.fromisoformat(last_active)).days
            if last_active
            else 10**9
        )
        if branch == "feature":
            return (
                support["active_days"] >= 30
                and support["span_days"] >= 45
                and support["observations"] >= 30
                and support["feature_coverage_ratio_at_20"] >= 0.9
                and stale_gap <= 30
            )
        return (
            support["observations"] >= 20
            and support["span_days"] >= 30
            and support["transitions"] >= 500
            and stale_gap <= 30
        )
    if level == "ROLE":
        if role == "UNKNOWN_ROLE":
            return False
        peer_users = support["users"] - int(user_id in support["user_ids"])
        if branch == "feature":
            return (
                peer_users >= 15
                and support["observations"] >= 300
                and support["feature_coverage_ratio_at_200"] >= 0.9
                and support["recent_30d_observations"] >= 100
            )
        return (
            peer_users >= 15
            and support["observations"] >= 300
            and support["transitions"] >= 10000
            and support["recent_30d_transitions"] >= 2000
        )
    if branch == "feature":
        return (
            support["users"] >= 200
            and support["observations"] >= 10000
            and support["feature_coverage_ratio_at_200"] >= 0.9
        )
    return (
        support["users"] >= 200
        and support["observations"] >= 10000
        and support["transitions"] >= 100000
    )


def _calibrate_row(row: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    calibrated: dict[str, float | None] = {"feature": None, "sequence": None}
    selected: dict[str, str] = {"feature": "NO_SCORE", "sequence": "NO_SCORE"}
    for branch, score_name in (
        ("feature", "feature_raw"),
        ("sequence", "sequence_raw"),
    ):
        raw_score = row[score_name]
        if raw_score is None:
            continue
        keys = (
            ("PERSON", str(row["role_epoch"])),
            ("ROLE", str(row["role"])),
            ("GLOBAL", "GLOBAL"),
        )
        for level, key in keys:
            entry = artifact["branches"][branch][level].get(key)
            if entry is None or not _ready(
                branch=branch,
                level=level,
                entry=entry,
                user_id=str(row["user_id"]),
                sequence_length=int(row["sequence_length"]),
                day=str(row["day"]),
                role=str(row["role"]),
            ):
                continue
            calibrated[branch] = calibrated_tail_score(float(raw_score), _stats(entry))
            selected[branch] = level
            break
    available = [value for value in calibrated.values() if value is not None]
    risk = fmean(available) if available else None
    return {
        **row,
        "feature_calibrated": calibrated["feature"],
        "sequence_calibrated": calibrated["sequence"],
        "feature_level": selected["feature"],
        "sequence_level": selected["sequence"],
        "risk": risk,
        "status": "SCORED" if risk is not None else "NO_SCORE",
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "user_id",
        "day",
        "split",
        "risk",
        "status",
        "feature_level",
        "sequence_level",
        "feature_raw",
        "sequence_raw",
        "feature_calibrated",
        "sequence_calibrated",
        "sequence_length",
        "role",
        "role_epoch",
        "active",
        "feature_coverage",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.store and args.split is None:
        raise ValueError("--store requires --split")
    if args.input and args.split is not None:
        raise ValueError("--split is only used with --store")
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != "tcn-transformer-ae.v4":
        raise ValueError("checkpoint model schema is not tcn-transformer-ae.v4")
    if args.input:
        windows = load_prepared_windows(args.input)
        directory = LdapDirectory.from_raw(args.raw_root.resolve())
        raw_rows = _score_rows(
            windows=windows,
            checkpoint=checkpoint,
            batch_size=args.batch_size,
            device=device,
            directory=directory,
        )
        preprocessing_checksum = str(windows.preprocessing_checksum.item())
        scaler_checksum = str(windows.scaler_checksum.item())
        input_kind = "npz"
    else:
        dataset = SQLiteWindowDataset(
            args.store,
            split=args.split,
            max_samples=args.max_samples,
        )
        raw_rows = _score_store_rows(
            dataset=dataset,
            checkpoint=checkpoint,
            batch_size=args.batch_size,
            device=device,
        )
        preprocessing_checksum = str(dataset.preprocessing_checksum)
        scaler_checksum = str(dataset.scaler_checksum)
        input_kind = "sqlite_user_day_store"
        dataset.close()
    checkpoint_preprocessing = checkpoint.get("preprocessing_checksum")
    checkpoint_scaler = checkpoint.get("scaler_checksum")
    if (
        checkpoint_preprocessing is not None
        and checkpoint_preprocessing != preprocessing_checksum
    ):
        raise ValueError("checkpoint and input preprocessing checksums do not match")
    if checkpoint_scaler is not None and checkpoint_scaler != scaler_checksum:
        raise ValueError("checkpoint and input scaler checksums do not match")
    reference_artifact = None
    if args.fit_reference_out:
        reference_artifact = _fit_reference_artifact(
            raw_rows,
            checkpoint_path=args.checkpoint,
            preprocessing_checksum=preprocessing_checksum,
            scaler_checksum=scaler_checksum,
        )
        args.fit_reference_out.parent.mkdir(parents=True, exist_ok=True)
        args.fit_reference_out.write_text(
            json.dumps(reference_artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.reference_in:
        reference_artifact = json.loads(args.reference_in.read_text(encoding="utf-8"))
        if (
            reference_artifact.get("schema_version") != "hierarchical-reference.v1"
            or reference_artifact.get("fit_split") != "TRAIN"
            or reference_artifact.get("frozen_after_train") is not True
        ):
            raise ValueError("reference artifact is not a frozen Train reference")
        if reference_artifact["checkpoint_sha256"] != _sha256(args.checkpoint):
            raise ValueError("reference artifact was fitted with a different checkpoint")
        if reference_artifact["preprocessing_checksum"] != preprocessing_checksum:
            raise ValueError("reference and input preprocessing checksums do not match")
        if reference_artifact["scaler_checksum"] != scaler_checksum:
            raise ValueError("reference and input scaler checksums do not match")
    rows = (
        [_calibrate_row(row, reference_artifact) for row in raw_rows]
        if args.reference_in
        else [
            {
                **row,
                "risk": None,
                "status": "RAW_ONLY",
                "feature_level": "NO_SCORE",
                "sequence_level": "NO_SCORE",
                "feature_calibrated": None,
                "sequence_calibrated": None,
            }
            for row in raw_rows
        ]
    )
    _write_rows(args.output, rows)
    result = {
        "schema_version": "score-run.v1",
        "samples": len(rows),
        "scored": sum(row["risk"] is not None for row in rows),
        "raw_feature_scores": sum(row["feature_raw"] is not None for row in rows),
        "raw_sequence_scores": sum(row["sequence_raw"] is not None for row in rows),
        "last_day_only": True,
        "reference_fitted": args.fit_reference_out is not None,
        "reference_applied": args.reference_in is not None,
        "output": str(args.output.resolve()),
        "device": device_name,
        "input_kind": input_kind,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "reference_sha256": (
            _sha256(args.reference_in) if args.reference_in else None
        ),
        "preprocessing_checksum": preprocessing_checksum,
        "scaler_checksum": scaler_checksum,
    }
    result["output_sha256"] = _sha256(args.output)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["manifest"] = str(manifest_path.resolve())
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
