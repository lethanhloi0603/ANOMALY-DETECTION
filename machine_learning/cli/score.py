"""Run last-day inference and frozen Person -> Role -> Global calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from datetime import date
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from insider_ml.artifacts import atomic_write_csv, atomic_write_json
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
    parser.add_argument(
        "--framework-config",
        type=Path,
        default=root / "backend" / "config" / "framework.v5.json",
    )
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


def _load_framework_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load framework config: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("framework config root must be an object")
    if payload.get("schema_version") != "framework.v5":
        raise ValueError("scoring requires an explicit framework.v5 config")
    return payload


def _validate_reference_artifact_contract(
    artifact: dict[str, Any],
    *,
    framework_config_checksum: str,
) -> None:
    schema_version = artifact.get("schema_version")
    if schema_version == "hierarchical-reference.v1":
        raise ValueError(
            "hierarchical-reference.v1 is a legacy framework.v4 artifact; "
            "refit it with framework.v5 instead of reinterpreting its readiness"
        )
    if (
        schema_version != "hierarchical-reference.v2"
        or artifact.get("fit_split") != "TRAIN"
        or artifact.get("frozen_after_train") is not True
    ):
        raise ValueError("reference artifact is not a frozen Train v2 reference")
    if artifact.get("framework_schema_version") != "framework.v5":
        raise ValueError("reference artifact was not fitted with framework.v5")
    if artifact.get("framework_config_sha256") != framework_config_checksum:
        raise ValueError("reference and scoring framework configs do not match")
    embedded_config = artifact.get("framework_config")
    if not isinstance(embedded_config, dict):
        raise ValueError("reference artifact does not embed its framework config")
    if embedded_config.get("schema_version") != "framework.v5":
        raise ValueError("embedded reference config is not framework.v5")


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
) -> Iterator[dict[str, Any]]:
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
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
                yield {
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
            offset += feature.shape[0]


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
    framework_config: dict[str, Any],
    framework_config_checksum: str,
) -> dict[str, Any]:
    if {row["split"] for row in rows} != {"TRAIN"}:
        raise ValueError("reference fitting accepts Train rows only")
    artifact: dict[str, Any] = {
        "schema_version": "hierarchical-reference.v2",
        "framework_schema_version": framework_config["schema_version"],
        "framework_config_sha256": framework_config_checksum,
        "framework_config": framework_config,
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
            if row[score_name] is None or (
                branch == "sequence" and int(row["sequence_length"]) < 2
            ):
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
                train_end = date.fromisoformat(
                    framework_config["splits"]["train"]["end"]
                )
                recent_rows = [
                    row
                    for row in scope_rows
                    if 0
                    <= (train_end - date.fromisoformat(str(row["day"]))).days
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
                        "feature_observation_days": len(scope_rows),
                        "observations_by_user": {
                            user: sum(
                                str(row["user_id"]) == user for row in scope_rows
                            )
                            for user in users
                        },
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
                        "transitions_by_user": {
                            user: sum(
                                max(int(row["sequence_length"]) - 1, 0)
                                for row in scope_rows
                                if str(row["user_id"]) == user
                            )
                            for user in users
                        },
                        "sequence_days": sum(
                            int(row["sequence_length"]) >= 2 for row in scope_rows
                        ),
                        "sequence_days_by_user": {
                            user: sum(
                                int(row["sequence_length"]) >= 2
                                for row in scope_rows
                                if str(row["user_id"]) == user
                            )
                            for user in users
                        },
                        "active_days": sum(bool(row["active"]) for row in scope_rows),
                        "mean_feature_coverage": fmean(
                            float(row["feature_coverage"]) for row in scope_rows
                        ),
                        "feature_coverage_ratio_at_20": float(
                            np.mean(feature_support >= 20)
                        ),
                        "feature_coverage_ratio_at_40": float(
                            np.mean(feature_support >= 40)
                        ),
                        "feature_coverage_ratio_at_200": float(
                            np.mean(feature_support >= 200)
                        ),
                        "minimum_nonzero_feature_support": int(
                            feature_support[feature_support > 0].min()
                            if np.any(feature_support > 0)
                            else 0
                        ),
                        # Safe-update v5 must be able to extend a Feature
                        # reference without reconstructing historical masks.
                        # Persist the exact immutable per-feature support, not
                        # only its minimum/coverage summaries.
                        "feature_observation_counts": [
                            int(value) for value in feature_support.tolist()
                        ],
                        "last_active_day": max(
                            (
                                str(row["day"])
                                for row in scope_rows
                                if bool(row["active"])
                            ),
                            default=None,
                        ),
                        "last_sequence_day": max(
                            (
                                str(row["day"])
                                for row in scope_rows
                                if int(row["sequence_length"]) >= 2
                            ),
                            default=None,
                        ),
                        "recent_30d_observations": len(recent_rows),
                        "recent_30d_observations_by_user": {
                            user: sum(
                                str(row["user_id"]) == user for row in recent_rows
                            )
                            for user in users
                        },
                        "recent_30d_transitions": sum(
                            max(int(row["sequence_length"]) - 1, 0)
                            for row in recent_rows
                        ),
                        "recent_30d_transitions_by_user": {
                            user: sum(
                                max(int(row["sequence_length"]) - 1, 0)
                                for row in recent_rows
                                if str(row["user_id"]) == user
                            )
                            for user in users
                        },
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
    split: str,
    role: str,
    framework_config: dict[str, Any],
) -> bool:
    support = entry["support"]
    readiness = framework_config["readiness"]
    branch_config = readiness[branch]
    if branch == "sequence" and sequence_length < int(
        branch_config["current_day"]["min_seq_len"]
    ):
        return False

    score_day = date.fromisoformat(day)
    primary = framework_config["primary_evaluation"]
    if (
        split in {"VALIDATION", "TEST"}
        and primary["readiness_mode"] == "FROZEN_AT_TRAIN_END"
    ):
        readiness_day = date.fromisoformat(framework_config["splits"]["train"]["end"])
    else:
        readiness_day = score_day

    if level == "PERSON":
        person_config = branch_config["person"]
        last_active_key = (
            "last_active_day" if branch == "feature" else "last_sequence_day"
        )
        last_active = support.get(last_active_key)
        stale_gap = (
            (readiness_day - date.fromisoformat(last_active)).days
            if last_active
            else 10**9
        )
        if branch == "feature":
            return (
                support["active_days"] >= int(person_config["min_active_days"])
                and support["span_days"] >= int(person_config["min_span_days"])
                and support["active_days"]
                >= int(person_config["min_active_days_in_current_role"])
                and support["mean_feature_coverage"]
                >= float(person_config["min_feature_coverage_ratio"])
                and support["minimum_nonzero_feature_support"]
                >= int(person_config["min_observations_per_used_feature"])
                and stale_gap <= int(person_config["max_last_active_gap_days"])
            )
        return (
            support["sequence_days"] >= int(person_config["min_sequence_days"])
            and support["span_days"] >= int(person_config["min_span_days"])
            and support["sequence_days"]
            >= int(person_config["min_sequence_days_in_current_role"])
            and support["transitions"] >= int(person_config["min_transitions"])
            and stale_gap <= int(person_config["max_stale_gap_days"])
        )

    if level == "ROLE":
        role_config = branch_config["role"]
        if role_config.get("require_known_role", True) and role == "UNKNOWN_ROLE":
            return False
        peer_users = support["users"] - int(user_id in support["user_ids"])
        peer_observations = support["observations"] - int(
            support["observations_by_user"].get(user_id, 0)
        )
        if branch == "feature":
            recent_peer_observations = support["recent_30d_observations"] - int(
                support["recent_30d_observations_by_user"].get(user_id, 0)
            )
            return (
                peer_users
                >= int(role_config["min_peer_users_excluding_subject"])
                and peer_observations >= int(role_config["min_peer_user_days"])
                and support["feature_coverage_ratio_at_200"]
                >= float(role_config["min_feature_coverage_ratio"])
                and recent_peer_observations
                >= int(role_config["min_recent_30d_user_days"])
            )
        peer_sequence_days = support["sequence_days"] - int(
            support["sequence_days_by_user"].get(user_id, 0)
        )
        peer_transitions = support["transitions"] - int(
            support["transitions_by_user"].get(user_id, 0)
        )
        recent_peer_transitions = support["recent_30d_transitions"] - int(
            support["recent_30d_transitions_by_user"].get(user_id, 0)
        )
        return (
            peer_users >= int(role_config["min_peer_users"])
            and peer_sequence_days >= int(role_config["min_sequence_days"])
            and peer_transitions >= int(role_config["min_transitions"])
            and recent_peer_transitions
            >= int(role_config["min_recent_30d_transitions"])
        )

    global_config = branch_config["global"]
    if branch == "feature":
        return (
            support["users"] >= int(global_config["min_users"])
            and support["observations"] >= int(global_config["min_user_days"])
            and support["feature_coverage_ratio_at_200"]
            >= float(global_config["min_feature_coverage_ratio"])
        )
    return (
        support["users"] >= int(global_config["min_users"])
        and support["sequence_days"] >= int(global_config["min_sequence_days"])
        and support["transitions"] >= int(global_config["min_transitions"])
    )


def _calibrate_row(
    row: dict[str, Any],
    artifact: dict[str, Any],
    framework_config: dict[str, Any],
) -> dict[str, Any]:
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
                split=str(row["split"]).upper(),
                role=str(row["role"]),
                framework_config=framework_config,
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


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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
    atomic_write_csv(path, fieldnames=fieldnames, rows=rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.store and args.split is None:
        raise ValueError("--store requires --split")
    if args.input and args.split is not None:
        raise ValueError("--split is only used with --store")
    framework_config = _load_framework_config(args.framework_config)
    framework_config_checksum = _sha256(args.framework_config)
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
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
        raw_rows: Iterable[dict[str, Any]] = _score_store_rows(
            dataset=dataset,
            checkpoint=checkpoint,
            batch_size=args.batch_size,
            device=device,
        )
        preprocessing_checksum = str(dataset.preprocessing_checksum)
        scaler_checksum = str(dataset.scaler_checksum)
        input_kind = "sqlite_user_day_store"
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
        raw_rows = list(raw_rows)
        reference_artifact = _fit_reference_artifact(
            raw_rows,
            checkpoint_path=args.checkpoint,
            preprocessing_checksum=preprocessing_checksum,
            scaler_checksum=scaler_checksum,
            framework_config=framework_config,
            framework_config_checksum=framework_config_checksum,
        )
        atomic_write_json(args.fit_reference_out, reference_artifact)
    if args.reference_in:
        reference_artifact = json.loads(args.reference_in.read_text(encoding="utf-8"))
        _validate_reference_artifact_contract(
            reference_artifact,
            framework_config_checksum=framework_config_checksum,
        )
        if reference_artifact["checkpoint_sha256"] != _sha256(args.checkpoint):
            raise ValueError("reference artifact was fitted with a different checkpoint")
        if reference_artifact["preprocessing_checksum"] != preprocessing_checksum:
            raise ValueError("reference and input preprocessing checksums do not match")
        if reference_artifact["scaler_checksum"] != scaler_checksum:
            raise ValueError("reference and input scaler checksums do not match")
    rows: Iterable[dict[str, Any]] = (
        (
            _calibrate_row(
                row,
                reference_artifact,
                reference_artifact["framework_config"],
            )
            for row in raw_rows
        )
        if args.reference_in
        else (
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
        )
    )
    counters = {
        "samples": 0,
        "scored": 0,
        "raw_feature_scores": 0,
        "raw_sequence_scores": 0,
    }

    def tracked_rows() -> Iterator[dict[str, Any]]:
        for row in rows:
            counters["samples"] += 1
            counters["scored"] += int(row["risk"] is not None)
            counters["raw_feature_scores"] += int(row["feature_raw"] is not None)
            counters["raw_sequence_scores"] += int(row["sequence_raw"] is not None)
            yield row

    try:
        _write_rows(args.output, tracked_rows())
    finally:
        if not args.input:
            dataset.close()
    result = {
        "schema_version": "score-run.v1",
        **counters,
        "last_day_only": True,
        "reference_fitted": args.fit_reference_out is not None,
        "reference_applied": args.reference_in is not None,
        "output": str(args.output.resolve()),
        "device": device_name,
        "input_kind": input_kind,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "framework_schema_version": framework_config["schema_version"],
        "framework_config_sha256": framework_config_checksum,
        "reference_sha256": (
            _sha256(args.reference_in) if args.reference_in else None
        ),
        "preprocessing_checksum": preprocessing_checksum,
        "scaler_checksum": scaler_checksum,
    }
    result["output_sha256"] = _sha256(args.output)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    atomic_write_json(manifest_path, result)
    result["manifest"] = str(manifest_path.resolve())
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
