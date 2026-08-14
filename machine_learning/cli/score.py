"""Run last-day inference and frozen Person -> Role -> Global calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from insider_ml.artifacts import atomic_write_csv, atomic_write_json
from insider_ml.cert_data import (
    PRIMARY_DISABLED_FEATURES,
    SOURCE_FILES,
    LdapDirectory,
    load_feature_names,
)
from insider_ml.contracts import TRAIN_END, TRAIN_START
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
from insider_ml.stream_store import connect_store, metadata_get

FRAMEWORK_V5 = "framework.v5"
FRAMEWORK_V6 = "framework.v6"
SUPPORTED_FRAMEWORK_VERSIONS = {FRAMEWORK_V5, FRAMEWORK_V6}
ENTRY_CONTRACT_VERSION = "hierarchical-reference-entry.v1"
ENTRY_CONTRACT_VERSION_V2 = "hierarchical-reference-entry.v2"
REFERENCE_SCHEMA_BY_FRAMEWORK = {
    FRAMEWORK_V5: "hierarchical-reference.v2",
    FRAMEWORK_V6: "hierarchical-reference.v3",
}
ENTRY_CONTRACT_BY_FRAMEWORK = {
    FRAMEWORK_V5: ENTRY_CONTRACT_VERSION,
    FRAMEWORK_V6: ENTRY_CONTRACT_VERSION_V2,
}
_ENTRY_CHECKSUM_FIELDS = ("identity", "stats", "support", "calibrator")
REFERENCE_ENDPOINT_POLICY = "ALL"
FEATURE_SCHEMA_VERSION = "feature128.v5"
SEQUENCE_SCHEMA_VERSION = "sequence7.v4"
FEATURE_VALUE_SPACE = "robust_scaled"


def _framework_version(framework_config: Mapping[str, Any]) -> str:
    version = framework_config.get("schema_version")
    if version not in SUPPORTED_FRAMEWORK_VERSIONS:
        raise ValueError("unsupported framework schema version")
    return str(version)


def _v6_enabled_feature_indices(
    framework_config: Mapping[str, Any],
) -> tuple[int, ...]:
    """Resolve the v6 readiness denominator from the locked Feature128 catalog."""

    try:
        contract = framework_config["readiness"]["feature"]["coverage_contract"]
        disabled_names = contract["disabled_feature_names"]
        configured_count = contract["enabled_feature_count"]
        minimum_ratio = contract["minimum_enabled_feature_ratio"]
    except (KeyError, TypeError) as exc:
        raise ValueError("framework.v6 has no enabled-feature coverage contract") from exc
    if (
        not isinstance(disabled_names, list)
        or set(disabled_names) != set(PRIMARY_DISABLED_FEATURES)
        or len(disabled_names) != len(PRIMARY_DISABLED_FEATURES)
        or isinstance(configured_count, bool)
        or not isinstance(configured_count, int)
        or isinstance(minimum_ratio, bool)
        or not isinstance(minimum_ratio, (int, float))
        or not 0.0 < float(minimum_ratio) <= 1.0
    ):
        raise ValueError("framework.v6 enabled-feature coverage contract is invalid")
    catalog_path = Path(__file__).resolve().parents[1] / "config" / "feature128.v5.json"
    feature_names = load_feature_names(catalog_path)
    enabled = tuple(
        index for index, name in enumerate(feature_names) if name not in disabled_names
    )
    if len(enabled) != configured_count or configured_count != 124:
        raise ValueError("framework.v6 must contain exactly 124 primary-enabled features")
    return enabled


def _reference_versions(
    framework_config: Mapping[str, Any],
) -> tuple[str, str]:
    version = _framework_version(framework_config)
    return REFERENCE_SCHEMA_BY_FRAMEWORK[version], ENTRY_CONTRACT_BY_FRAMEWORK[version]


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
        default=root / "backend" / "config" / "framework.v6.json",
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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_sha256(payload: object) -> str:
    """Hash the exact JSON contract representation used by reference entries."""

    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("reference entry is not canonical finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _reference_entry_checksum(entry: dict[str, Any]) -> str:
    try:
        payload = {field: entry[field] for field in _ENTRY_CHECKSUM_FIELDS}
    except KeyError as exc:
        raise ValueError("reference entry is missing checksum material") from exc
    return _canonical_json_sha256(payload)


def _seal_reference_entry(entry: dict[str, Any]) -> dict[str, Any]:
    entry["entry_checksum_sha256"] = _reference_entry_checksum(entry)
    return entry


def _validate_complete_source_attestation(
    attestation: object,
    *,
    require_source_sha256: bool,
) -> dict[str, Any]:
    if (
        not isinstance(attestation, dict)
        or attestation.get("schema_version")
        != "reference-source-attestation.v1"
        or attestation.get("complete_train") is not True
        or not isinstance(attestation.get("source"), str)
        or not attestation["source"]
        or (
            require_source_sha256
            and not _is_sha256(attestation.get("source_sha256"))
        )
    ):
        raise ValueError("source attestation is not a complete Train contract")

    source_kind = attestation.get("kind")
    if source_kind == "sqlite_user_day_store":
        try:
            attested_end = date.fromisoformat(
                str(attestation["daily_materialization_end_day"])
            )
        except (KeyError, ValueError) as exc:
            raise ValueError("store attestation has an invalid end day") from exc
        attested_sources = attestation.get("sources")
        if (
            attested_end < TRAIN_END
            or not _is_sha256(attestation.get("daily_materialization_signature"))
            or not isinstance(attested_sources, dict)
            or set(attested_sources) != set(SOURCE_FILES)
            or any(
                not isinstance(completion, dict)
                or "max_rows" not in completion
                or completion["max_rows"] is not None
                or not isinstance(completion.get("fingerprint"), dict)
                for completion in attested_sources.values()
            )
        ):
            raise ValueError("store attestation is incomplete")
    elif source_kind == "prepared_windows_npz":
        if (
            attestation.get("start_day") != TRAIN_START.isoformat()
            or attestation.get("end_day") != TRAIN_END.isoformat()
            or attestation.get("endpoint_policy") != REFERENCE_ENDPOINT_POLICY
            or "max_users" not in attestation
            or attestation["max_users"] is not None
            or isinstance(attestation.get("selected_user_count"), bool)
            or not isinstance(attestation.get("selected_user_count"), int)
            or attestation["selected_user_count"] < 1
            or not _is_sha256(attestation.get("source_sha256"))
            or not _is_sha256(attestation.get("manifest_sha256"))
            or not isinstance(attestation.get("manifest"), str)
            or not attestation["manifest"]
            or isinstance(attestation.get("samples"), bool)
            or not isinstance(attestation.get("samples"), int)
            or attestation["samples"] < 1
        ):
            raise ValueError("NPZ attestation is incomplete")
    else:
        raise ValueError("source attestation kind is unsupported")
    return attestation


def _validate_checkpoint_training_contract(
    checkpoint: Mapping[str, Any],
    *,
    framework_config: Mapping[str, Any],
    framework_config_checksum: str,
) -> None:
    """Require a complete, frozen Train checkpoint for reference calibration."""

    if (
        checkpoint.get("fit_split") != "TRAIN"
        or checkpoint.get("frozen_after_train") is not True
    ):
        raise ValueError("reference calibration requires a frozen Train checkpoint")
    try:
        expected_endpoint_policy = str(
            framework_config["training_sampling"]["endpoint_policy"]
        ).upper()
    except (KeyError, TypeError) as exc:
        raise ValueError("framework config has no training sampling contract") from exc
    framework_version = framework_config.get("schema_version")
    if (
        framework_version not in SUPPORTED_FRAMEWORK_VERSIONS
        or expected_endpoint_policy != "WEEKLY_TRAIN"
        or framework_config["training_sampling"].get(
            "reference_fit_uses_all_train_endpoints"
        )
        is not True
        or checkpoint.get("framework_schema_version") != framework_version
        or checkpoint.get("framework_config_sha256") != framework_config_checksum
        or checkpoint.get("training_endpoint_policy") != expected_endpoint_policy
        or checkpoint.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
        or checkpoint.get("sequence_schema_version") != SEQUENCE_SCHEMA_VERSION
        or checkpoint.get("feature_value_space") != FEATURE_VALUE_SPACE
        or not _is_sha256(checkpoint.get("preprocessing_checksum"))
        or not _is_sha256(checkpoint.get("scaler_checksum"))
        or isinstance(checkpoint.get("training_samples"), bool)
        or not isinstance(checkpoint.get("training_samples"), int)
        or checkpoint["training_samples"] < 1
    ):
        raise ValueError(
            "reference calibration requires a locked framework.v5/framework.v6 "
            "training contract"
        )
    try:
        _validate_complete_source_attestation(
            checkpoint.get("training_source_attestation"),
            require_source_sha256=True,
        )
    except ValueError as exc:
        raise ValueError(
            "reference calibration requires a complete Train checkpoint source"
        ) from exc


def _load_framework_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load framework config: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("framework config root must be an object")
    if payload.get("schema_version") not in SUPPORTED_FRAMEWORK_VERSIONS:
        raise ValueError(
            "scoring requires an explicit supported framework.v5/framework.v6 config"
        )
    if payload["schema_version"] == FRAMEWORK_V6:
        _v6_enabled_feature_indices(payload)
    return payload


def _validate_store_reference_source(path: Path) -> dict[str, Any]:
    connection = connect_store(path, read_only=True)
    try:
        if metadata_get(connection, "daily_materialization_complete") != "true":
            raise ValueError(
                "reference fitting requires a completed daily tensor store"
            )
        raw_end_day = metadata_get(connection, "daily_materialization_end_day")
        try:
            end_day = date.fromisoformat(str(raw_end_day))
        except ValueError as exc:
            raise ValueError(
                "reference fitting store has no valid materialization end day"
            ) from exc
        if end_day < TRAIN_END:
            raise ValueError(
                "reference fitting store does not cover the locked Train end"
            )

        source_attestations: dict[str, dict[str, Any]] = {}
        for source in SOURCE_FILES:
            raw_completion = metadata_get(connection, f"source_complete_{source}")
            if raw_completion is None:
                raise ValueError(
                    f"reference fitting store is missing completion metadata for {source}"
                )
            try:
                completion = json.loads(raw_completion)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"reference fitting store has invalid completion metadata for {source}"
                ) from exc
            if not isinstance(completion, dict) or not isinstance(
                completion.get("fingerprint"), dict
            ):
                raise ValueError(
                    f"reference fitting store has invalid completion metadata for {source}"
                )
            if "max_rows" not in completion or completion["max_rows"] is not None:
                raise ValueError(
                    f"reference fitting refuses capped source {source}"
                )
            source_attestations[source] = completion

        materialization_signature = metadata_get(
            connection,
            "daily_materialization_signature",
        )
        if not _is_sha256(materialization_signature):
            raise ValueError(
                "reference fitting store has no valid materialization signature"
            )
    finally:
        connection.close()

    return {
        "schema_version": "reference-source-attestation.v1",
        "kind": "sqlite_user_day_store",
        "complete_train": True,
        "source": str(path.resolve()),
        "daily_materialization_end_day": end_day.isoformat(),
        "daily_materialization_signature": materialization_signature,
        "sources": source_attestations,
    }


def _validate_prepared_reference_source(
    path: Path,
    *,
    preprocessing_checksum: str,
    scaler_checksum: str,
    sample_count: int,
    feature_schema_version: str,
    sequence_schema_version: str,
) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "reference fitting from NPZ requires its preparation manifest beside the input"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("preparation manifest root must be an object")
    if manifest.get("schema_version") != "cert-preparation-manifest.v1":
        raise ValueError("reference fitting requires cert-preparation-manifest.v1")
    if manifest.get("split") != "TRAIN":
        raise ValueError("reference fitting preparation manifest must attest TRAIN")
    if (
        manifest.get("start_day") != TRAIN_START.isoformat()
        or manifest.get("end_day") != TRAIN_END.isoformat()
    ):
        raise ValueError(
            "reference fitting preparation manifest must cover the exact locked Train range"
        )
    if (
        "smoke_row_cap_per_source" not in manifest
        or manifest["smoke_row_cap_per_source"] is not None
    ):
        raise ValueError("reference fitting refuses a capped NPZ source")
    if "max_users" not in manifest or manifest["max_users"] is not None:
        raise ValueError("reference fitting refuses a user-capped NPZ source")
    users = manifest.get("users")
    selected_user_count = manifest.get("selected_user_count")
    if (
        not isinstance(users, list)
        or not users
        or any(not isinstance(user_id, str) or not user_id for user_id in users)
        or len(set(users)) != len(users)
        or isinstance(selected_user_count, bool)
        or not isinstance(selected_user_count, int)
        or selected_user_count != len(users)
    ):
        raise ValueError("preparation manifest has invalid selected-user evidence")
    if manifest.get("endpoint_policy") != REFERENCE_ENDPOINT_POLICY:
        raise ValueError(
            "reference fitting NPZ must contain all locked Train endpoints"
        )
    try:
        manifested_output = Path(str(manifest["output"])).resolve()
    except KeyError as exc:
        raise ValueError("preparation manifest does not identify its output") from exc
    if manifested_output != path.resolve():
        raise ValueError("preparation manifest output path does not match the NPZ input")
    input_checksum = _sha256(path)
    if manifest.get("output_sha256") != input_checksum:
        raise ValueError("preparation manifest output checksum does not match the NPZ input")
    expected_fields = {
        "preprocessing_checksum": preprocessing_checksum,
        "scaler_checksum": scaler_checksum,
        "samples": sample_count,
        "feature_schema_version": feature_schema_version,
        "sequence_schema_version": sequence_schema_version,
    }
    for field_name, expected in expected_fields.items():
        if manifest.get(field_name) != expected:
            raise ValueError(
                f"preparation manifest {field_name} does not match the NPZ input"
            )
    return {
        "schema_version": "reference-source-attestation.v1",
        "kind": "prepared_windows_npz",
        "complete_train": True,
        "source": str(path.resolve()),
        "source_sha256": input_checksum,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "start_day": TRAIN_START.isoformat(),
        "end_day": TRAIN_END.isoformat(),
        "endpoint_policy": REFERENCE_ENDPOINT_POLICY,
        "max_users": None,
        "selected_user_count": selected_user_count,
        "samples": sample_count,
    }


def _validate_reference_artifact_contract(
    artifact: dict[str, Any],
    *,
    framework_config: dict[str, Any],
    framework_config_checksum: str,
) -> None:
    schema_version = artifact.get("schema_version")
    if schema_version == "hierarchical-reference.v1":
        raise ValueError(
            "hierarchical-reference.v1 is a legacy framework.v4 artifact; "
            "refit it with framework.v5 instead of reinterpreting its readiness"
        )
    framework_version = _framework_version(framework_config)
    expected_reference_schema, expected_entry_contract = _reference_versions(
        framework_config
    )
    if (
        schema_version != expected_reference_schema
        or artifact.get("fit_split") != "TRAIN"
        or artifact.get("frozen_after_train") is not True
    ):
        raise ValueError(
            f"reference artifact is not a frozen Train {expected_reference_schema} reference"
        )
    if artifact.get("framework_schema_version") != framework_version:
        raise ValueError(
            f"reference artifact was not fitted with {framework_version}"
        )
    if artifact.get("framework_config_sha256") != framework_config_checksum:
        raise ValueError("reference and scoring framework configs do not match")
    embedded_config = artifact.get("framework_config")
    if not isinstance(embedded_config, dict):
        raise ValueError("reference artifact does not embed its framework config")
    if embedded_config.get("schema_version") != framework_version:
        raise ValueError("embedded reference config version does not match")
    if embedded_config != framework_config:
        raise ValueError(
            "embedded reference framework config does not match the scoring config"
        )
    try:
        _validate_complete_source_attestation(
            artifact.get("source_attestation"),
            require_source_sha256=False,
        )
    except ValueError as exc:
        raise ValueError(
            "reference artifact does not contain a complete Train source attestation"
        ) from exc

    if artifact.get("entry_contract_version") != expected_entry_contract:
        raise ValueError("reference artifact has no supported entry contract")
    _validate_reference_branches(artifact, framework_config)


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


def _standalone_personal_target(framework_config: dict[str, Any]) -> int:
    try:
        calibrator_config = framework_config["safe_personalized_update"][
            "personal_calibrator"
        ]
        target = calibrator_config["standalone_min_safe_scores"]
    except (KeyError, TypeError) as exc:
        raise ValueError("framework config has no Personal calibrator contract") from exc
    if (
        isinstance(target, bool)
        or not isinstance(target, int)
        or target != 200
        or calibrator_config.get("below_min_method") != "parent_shrunk_ecdf.v1"
    ):
        raise ValueError("framework config has an invalid Personal calibrator contract")
    return target


def _build_reference_entry(
    *,
    branch: str,
    level: str,
    scope_key: str,
    scope_rows: list[dict[str, Any]],
    score_name: str,
    train_end: date,
    framework_config: dict[str, Any],
) -> dict[str, Any]:
    scores = np.asarray([row[score_name] for row in scope_rows], dtype=np.float64)
    stats = fit_reference(level, scope_key, scores)
    days = sorted({str(row["day"]) for row in scope_rows})
    users = sorted({str(row["user_id"]) for row in scope_rows})
    roles = sorted({str(row["role"]) for row in scope_rows})
    if level == "PERSON":
        if len(users) != 1 or len(roles) != 1:
            raise ValueError("Personal reference scope must identify one user and role")
        subject_user_id: str | None = users[0]
        role: str | None = roles[0]
    elif level == "ROLE":
        if roles != [scope_key]:
            raise ValueError("Role reference scope does not match its rows")
        subject_user_id = None
        role = scope_key
    else:
        subject_user_id = None
        role = None

    recent_rows = [
        row
        for row in scope_rows
        if 0 <= (train_end - date.fromisoformat(str(row["day"]))).days <= 29
    ]
    feature_support = np.zeros(128, dtype=np.int64)
    for row in scope_rows:
        packed = np.frombuffer(row["_feature_mask"], dtype=np.uint8)
        feature_support += np.unpackbits(packed, bitorder="little")[:128]
    support = {
        "observations": len(scope_rows),
        "feature_observation_days": len(scope_rows),
        "observations_by_user": {
            user: sum(str(row["user_id"]) == user for row in scope_rows)
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
        "feature_coverage_ratio_at_20": float(np.mean(feature_support >= 20)),
        "feature_coverage_ratio_at_40": float(np.mean(feature_support >= 40)),
        "feature_coverage_ratio_at_200": float(np.mean(feature_support >= 200)),
        "minimum_nonzero_feature_support": int(
            feature_support[feature_support > 0].min()
            if np.any(feature_support > 0)
            else 0
        ),
        # Safe-update v5 must be able to extend a Feature reference without
        # reconstructing historical masks. Preserve the immutable denominator.
        "feature_observation_counts": [
            int(value) for value in feature_support.tolist()
        ],
        "last_active_day": max(
            (str(row["day"]) for row in scope_rows if bool(row["active"])),
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
            user: sum(str(row["user_id"]) == user for row in recent_rows)
            for user in users
        },
        "recent_30d_transitions": sum(
            max(int(row["sequence_length"]) - 1, 0) for row in recent_rows
        ),
        "recent_30d_transitions_by_user": {
            user: sum(
                max(int(row["sequence_length"]) - 1, 0)
                for row in recent_rows
                if str(row["user_id"]) == user
            )
            for user in users
        },
    }
    framework_version = _framework_version(framework_config)
    if framework_version == FRAMEWORK_V6:
        enabled_indices = _v6_enabled_feature_indices(framework_config)
        enabled_support = feature_support[np.asarray(enabled_indices, dtype=np.int64)]
        support.update(
            {
                "enabled_feature_count": len(enabled_indices),
                "enabled_feature_support_ratio_at_40": float(
                    np.mean(enabled_support >= 40)
                ),
                "enabled_feature_support_ratio_at_200": float(
                    np.mean(enabled_support >= 200)
                ),
            }
        )
    return {
        "entry_schema_version": ENTRY_CONTRACT_BY_FRAMEWORK[framework_version],
        "identity": {
            "branch": branch,
            "level": level,
            "scope_key": scope_key,
            "subject_user_id": subject_user_id,
            "role": role,
        },
        "stats": _reference_to_json(stats),
        "support": support,
        "calibrator": {"method": "empirical_cdf.v1"},
    }


def _select_personal_parent(
    *,
    branch: str,
    person_entry: dict[str, Any],
    levels: dict[str, dict[str, dict[str, Any]]],
    framework_config: dict[str, Any],
) -> tuple[str, str, dict[str, Any]] | None:
    identity = person_entry["identity"]
    subject_user_id = str(identity["subject_user_id"])
    role = str(identity["role"])
    try:
        parent_order = framework_config["safe_personalized_update"]["admission"][
            "parent_order"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("framework config has no Personal parent order") from exc
    if parent_order != ["ROLE", "GLOBAL"]:
        raise ValueError("framework config has an invalid Personal parent order")

    train_end = date.fromisoformat(framework_config["splits"]["train"]["end"])
    sequence_length = (
        int(framework_config["readiness"][branch]["current_day"]["min_seq_len"])
        if branch == "sequence"
        else 0
    )
    for level in parent_order:
        scope_key = role if level == "ROLE" else "GLOBAL"
        entry = levels[level].get(scope_key)
        if entry is not None and _ready(
            branch=branch,
            level=level,
            entry=entry,
            user_id=subject_user_id,
            sequence_length=sequence_length,
            day=(train_end + timedelta(days=1)).isoformat(),
            split="VALIDATION",
            role=role,
            framework_config=framework_config,
        ):
            return level, scope_key, entry
    return None


def _fit_reference_artifact(
    rows: list[dict[str, Any]],
    *,
    checkpoint_path: Path,
    preprocessing_checksum: str,
    scaler_checksum: str,
    framework_config: dict[str, Any],
    framework_config_checksum: str,
    source_attestation: dict[str, Any],
) -> dict[str, Any]:
    if {row["split"] for row in rows} != {"TRAIN"}:
        raise ValueError("reference fitting accepts Train rows only")
    framework_version = _framework_version(framework_config)
    reference_schema, entry_contract = _reference_versions(framework_config)
    build_diagnostics: dict[str, Any] = {
        "personal_references_skipped": {"feature": {}, "sequence": {}},
    }
    artifact: dict[str, Any] = {
        "schema_version": reference_schema,
        "entry_contract_version": entry_contract,
        "framework_schema_version": framework_config["schema_version"],
        "framework_config_sha256": framework_config_checksum,
        "framework_config": framework_config,
        "fit_split": "TRAIN",
        "frozen_after_train": True,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "preprocessing_checksum": preprocessing_checksum,
        "scaler_checksum": scaler_checksum,
        "source_attestation": source_attestation,
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
            raw_score = row[score_name]
            if (
                raw_score is None
                or not math.isfinite(float(raw_score))
                or (
                branch == "sequence" and int(row["sequence_length"]) < 2
                )
            ):
                continue
            grouped["PERSON"][str(row["role_epoch"])].append(row)
            grouped["ROLE"][str(row["role"])].append(row)
            grouped["GLOBAL"]["GLOBAL"].append(row)
        train_end = date.fromisoformat(framework_config["splits"]["train"]["end"])
        for level in ("ROLE", "GLOBAL"):
            scopes = grouped[level]
            for scope_key, scope_rows in scopes.items():
                entry = _build_reference_entry(
                    branch=branch,
                    level=level,
                    scope_key=scope_key,
                    scope_rows=scope_rows,
                    score_name=score_name,
                    train_end=train_end,
                    framework_config=framework_config,
                )
                levels[level][scope_key] = _seal_reference_entry(entry)

        target = _standalone_personal_target(framework_config)
        for scope_key, scope_rows in grouped["PERSON"].items():
            entry = _build_reference_entry(
                branch=branch,
                level="PERSON",
                scope_key=scope_key,
                scope_rows=scope_rows,
                score_name=score_name,
                train_end=train_end,
                framework_config=framework_config,
            )
            observation_count = int(entry["stats"]["observation_count"])
            if observation_count < target:
                parent = _select_personal_parent(
                    branch=branch,
                    person_entry=entry,
                    levels=levels,
                    framework_config=framework_config,
                )
                if parent is None:
                    build_diagnostics["personal_references_skipped"][branch][
                        "NO_READY_PARENT"
                    ] = (
                        build_diagnostics["personal_references_skipped"][branch].get(
                            "NO_READY_PARENT", 0
                        )
                        + 1
                    )
                    continue
                parent_level, parent_scope_key, parent_entry = parent
                entry["calibrator"] = {
                    "method": "parent_shrunk_ecdf.v1",
                    "standalone_target_observations": target,
                    "parent": {
                        "branch": branch,
                        "level": parent_level,
                        "scope_key": parent_scope_key,
                        "entry_checksum_sha256": parent_entry[
                            "entry_checksum_sha256"
                        ],
                    },
                }
            levels["PERSON"][scope_key] = _seal_reference_entry(entry)
        artifact["branches"][branch] = levels
    build_diagnostics["reference_entry_counts"] = {
        branch: {level: len(scopes) for level, scopes in levels.items()}
        for branch, levels in artifact["branches"].items()
    }
    artifact["build_diagnostics"] = build_diagnostics
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


def _readiness_reasons(
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
) -> tuple[str, ...]:
    support = entry["support"]
    readiness = framework_config["readiness"]
    branch_config = readiness[branch]
    if branch == "sequence" and sequence_length < int(
        branch_config["current_day"]["min_seq_len"]
    ):
        return ("CURRENT_SEQUENCE_TOO_SHORT",)

    score_day = date.fromisoformat(day)
    primary = framework_config["primary_evaluation"]
    if (
        split in {"VALIDATION", "TEST"}
        and primary["readiness_mode"] == "FROZEN_AT_TRAIN_END"
    ):
        readiness_day = date.fromisoformat(
            framework_config["splits"]["train"]["end"]
        ) + timedelta(days=1)
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
            reasons: list[str] = []
            if support["active_days"] < int(person_config["min_active_days"]):
                reasons.append("PERSON_ACTIVE_DAYS_LOW")
            if support["span_days"] < int(person_config["min_span_days"]):
                reasons.append("PERSON_SPAN_DAYS_LOW")
            if support["active_days"] < int(
                person_config["min_active_days_in_current_role"]
            ):
                reasons.append("PERSON_ROLE_ACTIVE_DAYS_LOW")
            if _framework_version(framework_config) == FRAMEWORK_V6:
                required_support = int(
                    person_config["min_observations_per_enabled_feature"]
                )
                if required_support != 40:
                    raise ValueError("framework.v6 Person Feature support must be 40")
                coverage = support["enabled_feature_support_ratio_at_40"]
                minimum_coverage = framework_config["readiness"]["feature"][
                    "coverage_contract"
                ]["minimum_enabled_feature_ratio"]
                if coverage < float(minimum_coverage):
                    reasons.append("PERSON_ENABLED_FEATURE_SUPPORT_RATIO_LOW")
            else:
                if support["mean_feature_coverage"] < float(
                    person_config["min_feature_coverage_ratio"]
                ):
                    reasons.append("PERSON_MEAN_FEATURE_COVERAGE_LOW")
                if support["minimum_nonzero_feature_support"] < int(
                    person_config["min_observations_per_used_feature"]
                ):
                    reasons.append("PERSON_FEATURE_SUPPORT_LOW")
            if stale_gap > int(person_config["max_last_active_gap_days"]):
                reasons.append("PERSON_STALE")
            return tuple(reasons)
        reasons = []
        if support["sequence_days"] < int(person_config["min_sequence_days"]):
            reasons.append("PERSON_SEQUENCE_DAYS_LOW")
        if support["span_days"] < int(person_config["min_span_days"]):
            reasons.append("PERSON_SPAN_DAYS_LOW")
        if support["sequence_days"] < int(
            person_config["min_sequence_days_in_current_role"]
        ):
            reasons.append("PERSON_ROLE_SEQUENCE_DAYS_LOW")
        if support["transitions"] < int(person_config["min_transitions"]):
            reasons.append("PERSON_TRANSITIONS_LOW")
        if stale_gap > int(person_config["max_stale_gap_days"]):
            reasons.append("PERSON_STALE")
        return tuple(reasons)

    if level == "ROLE":
        role_config = branch_config["role"]
        reasons = []
        if role_config.get("require_known_role", True) and role == "UNKNOWN_ROLE":
            reasons.append("ROLE_UNKNOWN")
        peer_users = support["users"] - int(user_id in support["user_ids"])
        peer_observations = support["observations"] - int(
            support["observations_by_user"].get(user_id, 0)
        )
        if branch == "feature":
            recent_peer_observations = support["recent_30d_observations"] - int(
                support["recent_30d_observations_by_user"].get(user_id, 0)
            )
            if peer_users < int(role_config["min_peer_users_excluding_subject"]):
                reasons.append("ROLE_PEER_USERS_LOW")
            if peer_observations < int(role_config["min_peer_user_days"]):
                reasons.append("ROLE_PEER_USER_DAYS_LOW")
            if recent_peer_observations < int(
                role_config["min_recent_30d_user_days"]
            ):
                reasons.append("ROLE_RECENT_USER_DAYS_LOW")
            if _framework_version(framework_config) == FRAMEWORK_V6:
                required_support = int(
                    role_config["min_support_per_enabled_feature"]
                )
                if required_support != 200:
                    raise ValueError("framework.v6 Role Feature support must be 200")
                coverage = support["enabled_feature_support_ratio_at_200"]
                minimum_coverage = framework_config["readiness"]["feature"][
                    "coverage_contract"
                ]["minimum_enabled_feature_ratio"]
                if coverage < float(minimum_coverage):
                    reasons.append("ROLE_ENABLED_FEATURE_SUPPORT_RATIO_LOW")
            else:
                if support["mean_feature_coverage"] < float(
                    role_config["min_feature_coverage_ratio"]
                ):
                    reasons.append("ROLE_MEAN_FEATURE_COVERAGE_LOW")
                if support["minimum_nonzero_feature_support"] < int(
                    role_config["min_support_per_feature"]
                ):
                    reasons.append("ROLE_FEATURE_SUPPORT_LOW")
            return tuple(reasons)
        peer_sequence_days = support["sequence_days"] - int(
            support["sequence_days_by_user"].get(user_id, 0)
        )
        peer_transitions = support["transitions"] - int(
            support["transitions_by_user"].get(user_id, 0)
        )
        recent_peer_transitions = support["recent_30d_transitions"] - int(
            support["recent_30d_transitions_by_user"].get(user_id, 0)
        )
        if peer_users < int(role_config["min_peer_users"]):
            reasons.append("ROLE_PEER_USERS_LOW")
        if peer_sequence_days < int(role_config["min_sequence_days"]):
            reasons.append("ROLE_SEQUENCE_DAYS_LOW")
        if peer_transitions < int(role_config["min_transitions"]):
            reasons.append("ROLE_TRANSITIONS_LOW")
        if recent_peer_transitions < int(
            role_config["min_recent_30d_transitions"]
        ):
            reasons.append("ROLE_RECENT_TRANSITIONS_LOW")
        return tuple(reasons)

    global_config = branch_config["global"]
    if branch == "feature":
        reasons = []
        if support["users"] < int(global_config["min_users"]):
            reasons.append("GLOBAL_USERS_LOW")
        if support["observations"] < int(global_config["min_user_days"]):
            reasons.append("GLOBAL_USER_DAYS_LOW")
        if _framework_version(framework_config) == FRAMEWORK_V6:
            required_support = int(
                global_config["min_support_per_enabled_feature"]
            )
            if required_support != 200:
                raise ValueError("framework.v6 Global Feature support must be 200")
            coverage = support["enabled_feature_support_ratio_at_200"]
            minimum_coverage = framework_config["readiness"]["feature"][
                "coverage_contract"
            ]["minimum_enabled_feature_ratio"]
            if coverage < float(minimum_coverage):
                reasons.append("GLOBAL_ENABLED_FEATURE_SUPPORT_RATIO_LOW")
        elif support["mean_feature_coverage"] < float(
            global_config["min_feature_coverage_ratio"]
        ):
            reasons.append("GLOBAL_MEAN_FEATURE_COVERAGE_LOW")
        return tuple(reasons)
    reasons = []
    if support["users"] < int(global_config["min_users"]):
        reasons.append("GLOBAL_USERS_LOW")
    if support["sequence_days"] < int(global_config["min_sequence_days"]):
        reasons.append("GLOBAL_SEQUENCE_DAYS_LOW")
    if support["transitions"] < int(global_config["min_transitions"]):
        reasons.append("GLOBAL_TRANSITIONS_LOW")
    return tuple(reasons)


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
    return not _readiness_reasons(
        branch=branch,
        level=level,
        entry=entry,
        user_id=user_id,
        sequence_length=sequence_length,
        day=day,
        split=split,
        role=role,
        framework_config=framework_config,
    )


_SUPPORT_FIELDS = {
    "observations",
    "feature_observation_days",
    "observations_by_user",
    "users",
    "user_ids",
    "span_days",
    "transitions",
    "transitions_by_user",
    "sequence_days",
    "sequence_days_by_user",
    "active_days",
    "mean_feature_coverage",
    "feature_coverage_ratio_at_20",
    "feature_coverage_ratio_at_40",
    "feature_coverage_ratio_at_200",
    "minimum_nonzero_feature_support",
    "feature_observation_counts",
    "last_active_day",
    "last_sequence_day",
    "recent_30d_observations",
    "recent_30d_observations_by_user",
    "recent_30d_transitions",
    "recent_30d_transitions_by_user",
}
_SUPPORT_FIELDS_V2 = _SUPPORT_FIELDS | {
    "enabled_feature_count",
    "enabled_feature_support_ratio_at_40",
    "enabled_feature_support_ratio_at_200",
}


def _is_nonnegative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _validate_support_day(value: object, *, required: bool, field_name: str) -> None:
    if value is None:
        if required:
            raise ValueError(f"reference support {field_name} is required")
        return
    if not required:
        raise ValueError(f"reference support {field_name} must be null")
    if not isinstance(value, str):
        raise ValueError(f"reference support {field_name} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"reference support {field_name} must be an ISO date"
        ) from exc
    if not TRAIN_START <= parsed <= TRAIN_END:
        raise ValueError(f"reference support {field_name} is outside locked Train")


def _validate_reference_support(
    support: object,
    *,
    observation_count: int,
    entry_contract_version: str,
) -> list[str]:
    expected_fields = (
        _SUPPORT_FIELDS_V2
        if entry_contract_version == ENTRY_CONTRACT_VERSION_V2
        else _SUPPORT_FIELDS
    )
    if not isinstance(support, dict) or set(support) != expected_fields:
        raise ValueError("reference entry support does not match its exact contract")

    integer_fields = (
        "observations",
        "feature_observation_days",
        "users",
        "span_days",
        "transitions",
        "sequence_days",
        "active_days",
        "minimum_nonzero_feature_support",
        "recent_30d_observations",
        "recent_30d_transitions",
    )
    if any(not _is_nonnegative_int(support[field]) for field in integer_fields):
        raise ValueError("reference support counters must be non-negative integers")
    if (
        support["observations"] != observation_count
        or support["feature_observation_days"] != observation_count
        or support["span_days"] < 1
        or support["span_days"] > (TRAIN_END - TRAIN_START).days + 1
        or support["active_days"] > observation_count
        or support["sequence_days"] > observation_count
        or support["recent_30d_observations"] > observation_count
        or support["recent_30d_transitions"] > support["transitions"]
    ):
        raise ValueError("reference support aggregate counters are inconsistent")

    user_ids = support["user_ids"]
    if (
        not isinstance(user_ids, list)
        or any(not isinstance(user, str) or not user for user in user_ids)
        or len(set(user_ids)) != len(user_ids)
        or support["users"] != len(user_ids)
        or not user_ids
    ):
        raise ValueError("reference entry user identities are inconsistent")
    observation_map = support["observations_by_user"]
    per_user_contracts = (
        ("observations_by_user", "observations", False),
        ("transitions_by_user", "transitions", True),
        ("sequence_days_by_user", "sequence_days", True),
        (
            "recent_30d_observations_by_user",
            "recent_30d_observations",
            True,
        ),
        ("recent_30d_transitions_by_user", "recent_30d_transitions", True),
    )
    for map_name, total_name, allow_zero in per_user_contracts:
        per_user = support[map_name]
        if (
            not isinstance(per_user, dict)
            or set(per_user) != set(user_ids)
            or any(
                not _is_nonnegative_int(value) or (not allow_zero and value < 1)
                for value in per_user.values()
            )
            or sum(per_user.values()) != support[total_name]
        ):
            raise ValueError(f"reference support {map_name} is inconsistent")
    for user in user_ids:
        if (
            support["sequence_days_by_user"][user] > observation_map[user]
            or support["recent_30d_observations_by_user"][user]
            > observation_map[user]
            or support["recent_30d_transitions_by_user"][user]
            > support["transitions_by_user"][user]
        ):
            raise ValueError("reference per-user day support is inconsistent")

    coverage_fields = (
        "mean_feature_coverage",
        "feature_coverage_ratio_at_20",
        "feature_coverage_ratio_at_40",
        "feature_coverage_ratio_at_200",
    )
    if any(
        isinstance(support[field], bool)
        or not isinstance(support[field], (int, float))
        or not math.isfinite(float(support[field]))
        or not 0.0 <= float(support[field]) <= 1.0
        for field in coverage_fields
    ):
        raise ValueError("reference feature coverage support must be finite ratios")
    feature_counts = support["feature_observation_counts"]
    if (
        not isinstance(feature_counts, list)
        or len(feature_counts) != 128
        or any(
            not _is_nonnegative_int(value) or value > observation_count
            for value in feature_counts
        )
    ):
        raise ValueError("reference feature support must contain 128 valid counters")
    nonzero_counts = [value for value in feature_counts if value > 0]
    expected_minimum = min(nonzero_counts, default=0)
    if support["minimum_nonzero_feature_support"] != expected_minimum:
        raise ValueError("reference minimum feature support is inconsistent")
    expected_mean_coverage = sum(feature_counts) / (observation_count * 128)
    if not math.isclose(
        float(support["mean_feature_coverage"]),
        expected_mean_coverage,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("reference support mean feature coverage is inconsistent")
    for threshold, field_name in (
        (20, "feature_coverage_ratio_at_20"),
        (40, "feature_coverage_ratio_at_40"),
        (200, "feature_coverage_ratio_at_200"),
    ):
        expected_ratio = sum(value >= threshold for value in feature_counts) / 128
        if not math.isclose(
            float(support[field_name]),
            expected_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"reference {field_name} is inconsistent")

    if entry_contract_version == ENTRY_CONTRACT_VERSION_V2:
        enabled_indices = tuple(
            index
            for index in range(128)
            if index not in {115, 116, 117, 118}
        )
        if support["enabled_feature_count"] != len(enabled_indices):
            raise ValueError("reference enabled feature count is inconsistent")
        enabled_counts = [feature_counts[index] for index in enabled_indices]
        for threshold, field_name in (
            (40, "enabled_feature_support_ratio_at_40"),
            (200, "enabled_feature_support_ratio_at_200"),
        ):
            value = support[field_name]
            expected_ratio = sum(count >= threshold for count in enabled_counts) / len(
                enabled_counts
            )
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not math.isclose(
                    float(value),
                    expected_ratio,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(f"reference {field_name} is inconsistent")

    _validate_support_day(
        support["last_active_day"],
        required=support["active_days"] > 0,
        field_name="last_active_day",
    )
    _validate_support_day(
        support["last_sequence_day"],
        required=support["sequence_days"] > 0,
        field_name="last_sequence_day",
    )
    return user_ids


def _validate_reference_entry_base(
    *,
    branch: str,
    level: str,
    scope_key: str,
    entry: object,
    entry_contract_version: str,
) -> dict[str, Any]:
    required_entry_fields = {
        "entry_schema_version",
        "identity",
        "stats",
        "support",
        "calibrator",
        "entry_checksum_sha256",
    }
    if not isinstance(entry, dict) or set(entry) != required_entry_fields:
        raise ValueError("reference entry does not match the exact v1 contract")
    if entry.get("entry_schema_version") != entry_contract_version:
        raise ValueError("reference entry schema version is unsupported")

    identity = entry.get("identity")
    identity_fields = {
        "branch",
        "level",
        "scope_key",
        "subject_user_id",
        "role",
    }
    if not isinstance(identity, dict) or set(identity) != identity_fields:
        raise ValueError("reference entry identity contract is invalid")
    if (
        identity.get("branch") != branch
        or identity.get("level") != level
        or identity.get("scope_key") != scope_key
    ):
        raise ValueError("reference entry identity does not match its artifact path")
    subject_user_id = identity.get("subject_user_id")
    identity_role = identity.get("role")
    if level == "PERSON":
        if (
            not isinstance(subject_user_id, str)
            or not subject_user_id
            or not isinstance(identity_role, str)
            or not identity_role
        ):
            raise ValueError("Personal reference entry identity is incomplete")
    elif level == "ROLE":
        if subject_user_id is not None or identity_role != scope_key or not scope_key:
            raise ValueError("Role reference entry identity is invalid")
    elif (
        scope_key != "GLOBAL"
        or subject_user_id is not None
        or identity_role is not None
    ):
        raise ValueError("Global reference entry identity is invalid")

    stats = entry.get("stats")
    stats_fields = {
        "level",
        "scope_key",
        "location",
        "scale",
        "observation_count",
        "sorted_scores",
    }
    if not isinstance(stats, dict) or set(stats) != stats_fields:
        raise ValueError("reference entry statistics contract is invalid")
    if stats.get("level") != level or stats.get("scope_key") != scope_key:
        raise ValueError("reference statistics do not match their artifact path")
    location = stats.get("location")
    scale = stats.get("scale")
    if (
        isinstance(location, bool)
        or not isinstance(location, (int, float))
        or not math.isfinite(float(location))
        or isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
    ):
        raise ValueError("reference statistics must have a finite location and scale")
    observation_count = stats.get("observation_count")
    sorted_scores = stats.get("sorted_scores")
    if (
        isinstance(observation_count, bool)
        or not isinstance(observation_count, int)
        or observation_count < 1
        or not isinstance(sorted_scores, list)
        or observation_count != len(sorted_scores)
    ):
        raise ValueError("reference observation count does not match sorted scores")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in sorted_scores
    ):
        raise ValueError("reference sorted scores must all be finite numbers")
    if any(
        float(left) > float(right)
        for left, right in zip(sorted_scores, sorted_scores[1:], strict=False)
    ):
        raise ValueError("reference scores are not sorted ascending")

    user_ids = _validate_reference_support(
        entry.get("support"),
        observation_count=observation_count,
        entry_contract_version=entry_contract_version,
    )
    if level == "PERSON" and user_ids != [subject_user_id]:
        raise ValueError("Personal reference subject does not match its support")

    checksum = entry.get("entry_checksum_sha256")
    if not _is_sha256(checksum) or checksum != _reference_entry_checksum(entry):
        raise ValueError("reference entry checksum does not match its contents")
    return entry


def _validate_reference_branches(
    artifact: dict[str, Any],
    framework_config: dict[str, Any],
) -> None:
    _reference_schema, entry_contract_version = _reference_versions(framework_config)
    if artifact.get("entry_contract_version") != entry_contract_version:
        raise ValueError("reference artifact has no supported entry contract")
    branches = artifact.get("branches")
    if not isinstance(branches, dict) or set(branches) != {"feature", "sequence"}:
        raise ValueError("reference artifact must contain exact feature/sequence branches")

    validated: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for branch in ("feature", "sequence"):
        levels = branches[branch]
        if not isinstance(levels, dict) or set(levels) != {
            "PERSON",
            "ROLE",
            "GLOBAL",
        }:
            raise ValueError("reference branch must contain exact hierarchy levels")
        validated[branch] = {}
        for level in ("PERSON", "ROLE", "GLOBAL"):
            scopes = levels[level]
            if not isinstance(scopes, dict):
                raise ValueError("reference hierarchy level must be an object")
            if level == "GLOBAL" and not set(scopes).issubset({"GLOBAL"}):
                raise ValueError("Global reference entries must use the GLOBAL path")
            validated[branch][level] = {}
            for scope_key, raw_entry in scopes.items():
                if not isinstance(scope_key, str) or not scope_key:
                    raise ValueError("reference scope key must be a non-empty string")
                validated[branch][level][scope_key] = _validate_reference_entry_base(
                    branch=branch,
                    level=level,
                    scope_key=scope_key,
                    entry=raw_entry,
                    entry_contract_version=entry_contract_version,
                )

    target = _standalone_personal_target(framework_config)
    for branch, levels in validated.items():
        for level in ("ROLE", "GLOBAL"):
            for entry in levels[level].values():
                if entry["calibrator"] != {"method": "empirical_cdf.v1"}:
                    raise ValueError("Role/Global references require empirical CDF")

        for entry in levels["PERSON"].values():
            count = int(entry["stats"]["observation_count"])
            calibrator = entry["calibrator"]
            if count >= target:
                if calibrator != {"method": "empirical_cdf.v1"}:
                    raise ValueError(
                        "standalone Personal reference requires empirical CDF"
                    )
                continue
            if not isinstance(calibrator, dict) or set(calibrator) != {
                "method",
                "standalone_target_observations",
                "parent",
            }:
                raise ValueError("small Personal reference requires a pinned parent")
            if (
                calibrator.get("method") != "parent_shrunk_ecdf.v1"
                or calibrator.get("standalone_target_observations") != target
            ):
                raise ValueError("Personal parent-shrunk calibrator contract is invalid")
            parent = calibrator.get("parent")
            if not isinstance(parent, dict) or set(parent) != {
                "branch",
                "level",
                "scope_key",
                "entry_checksum_sha256",
            }:
                raise ValueError("Personal calibration parent contract is invalid")
            parent_level = parent.get("level")
            parent_scope_key = parent.get("scope_key")
            if (
                parent.get("branch") != branch
                or parent_level not in {"ROLE", "GLOBAL"}
                or not isinstance(parent_scope_key, str)
            ):
                raise ValueError("Personal calibration parent path is invalid")
            parent_entry = levels[parent_level].get(parent_scope_key)
            if (
                parent_entry is None
                or parent.get("entry_checksum_sha256")
                != parent_entry["entry_checksum_sha256"]
            ):
                raise ValueError("Personal calibration parent checksum is invalid")
            try:
                expected_parent = _select_personal_parent(
                    branch=branch,
                    person_entry=entry,
                    levels=levels,
                    framework_config=framework_config,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("reference parent readiness contract is invalid") from exc
            if expected_parent is None:
                raise ValueError("small Personal reference has no ready parent")
            expected_level, expected_scope_key, expected_entry = expected_parent
            if (
                parent_level != expected_level
                or parent_scope_key != expected_scope_key
                or parent.get("entry_checksum_sha256")
                != expected_entry["entry_checksum_sha256"]
            ):
                raise ValueError("Personal reference does not pin the expected parent")


def _calibrated_entry_score(
    raw_score: float,
    entry: dict[str, Any],
    branch_levels: dict[str, dict[str, dict[str, Any]]],
) -> float:
    calibrator = entry["calibrator"]
    method = calibrator.get("method")
    personal_q = calibrated_tail_score(raw_score, _stats(entry))
    if method == "empirical_cdf.v1":
        return personal_q
    if method != "parent_shrunk_ecdf.v1":
        raise ValueError("reference entry has an unsupported calibrator")
    parent = calibrator["parent"]
    parent_entry = branch_levels[parent["level"]].get(parent["scope_key"])
    if (
        parent_entry is None
        or parent.get("branch") != entry["identity"]["branch"]
        or parent.get("entry_checksum_sha256")
        != parent_entry.get("entry_checksum_sha256")
    ):
        raise ValueError("parent-shrunk reference cannot resolve its pinned parent")
    target = calibrator["standalone_target_observations"]
    count = int(entry["stats"]["observation_count"])
    if isinstance(target, bool) or not isinstance(target, int) or not 0 < count < target:
        raise ValueError("parent-shrunk reference has invalid support")
    parent_q = calibrated_tail_score(raw_score, _stats(parent_entry))
    alpha = count / target
    return alpha * personal_q + (1.0 - alpha) * parent_q


def _calibrate_row(
    row: dict[str, Any],
    artifact: dict[str, Any],
    framework_config: dict[str, Any],
) -> dict[str, Any]:
    split = str(row["split"]).upper()
    score_day = date.fromisoformat(str(row["day"]))
    train_end = date.fromisoformat(framework_config["splits"]["train"]["end"])
    if split == "TRAIN" or score_day <= train_end:
        raise ValueError(
            "a frozen Train reference can only be applied to rows after Train end"
        )

    if (
        _framework_version(framework_config) == FRAMEWORK_V6
        and not bool(row.get("active", False))
    ):
        return {
            **row,
            "feature_calibrated": None,
            "sequence_calibrated": None,
            "feature_level": "NO_SCORE",
            "sequence_level": "NO_SCORE",
            "risk": None,
            "status": "NO_SCORE",
            "_readiness_trace": {
                "feature": ("CURRENT_DAY_INACTIVE",),
                "sequence": ("CURRENT_DAY_INACTIVE",),
            },
        }

    calibrated: dict[str, float | None] = {"feature": None, "sequence": None}
    selected: dict[str, str] = {"feature": "NO_SCORE", "sequence": "NO_SCORE"}
    readiness_trace: dict[str, list[str]] = {"feature": [], "sequence": []}
    for branch, score_name in (
        ("feature", "feature_raw"),
        ("sequence", "sequence_raw"),
    ):
        raw_score = row[score_name]
        if raw_score is None:
            readiness_trace[branch].append("RAW_SCORE_MISSING")
            continue
        keys = (
            ("PERSON", str(row["role_epoch"])),
            ("ROLE", str(row["role"])),
            ("GLOBAL", "GLOBAL"),
        )
        for level, key in keys:
            entry = artifact["branches"][branch][level].get(key)
            if entry is None:
                readiness_trace[branch].append(f"{level}_REFERENCE_MISSING")
                continue
            reasons = _readiness_reasons(
                branch=branch,
                level=level,
                entry=entry,
                user_id=str(row["user_id"]),
                sequence_length=int(row["sequence_length"]),
                day=str(row["day"]),
                split=str(row["split"]).upper(),
                role=str(row["role"]),
                framework_config=framework_config,
            )
            if reasons:
                readiness_trace[branch].extend(reasons)
                continue
            calibrated[branch] = _calibrated_entry_score(
                float(raw_score),
                entry,
                artifact["branches"][branch],
            )
            selected[branch] = level
            readiness_trace[branch].append(f"SELECTED_{level}")
            break
        if selected[branch] == "NO_SCORE":
            readiness_trace[branch].append("NO_READY_REFERENCE")
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
        "_readiness_trace": {
            branch: tuple(reasons) for branch, reasons in readiness_trace.items()
        },
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
    if args.fit_reference_out and args.max_samples is not None:
        raise ValueError(
            "--fit-reference-out requires the complete Train split; "
            "--max-samples is smoke-only"
        )
    if args.reference_in and args.store and args.split == "TRAIN":
        raise ValueError(
            "a frozen Train reference cannot be applied back to the Train split"
        )
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
    if args.fit_reference_out or args.reference_in:
        _validate_checkpoint_training_contract(
            checkpoint,
            framework_config=framework_config,
            framework_config_checksum=framework_config_checksum,
        )
    source_attestation: dict[str, Any] | None = None
    if args.input:
        windows = load_prepared_windows(args.input)
        if args.fit_reference_out:
            source_attestation = _validate_prepared_reference_source(
                args.input,
                preprocessing_checksum=str(windows.preprocessing_checksum.item()),
                scaler_checksum=str(windows.scaler_checksum.item()),
                sample_count=int(windows.feature_values.shape[0]),
                feature_schema_version=str(windows.feature_schema_version.item()),
                sequence_schema_version=str(windows.sequence_schema_version.item()),
            )
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
        if args.fit_reference_out:
            source_attestation = _validate_store_reference_source(args.store)
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
        if source_attestation is None:
            raise RuntimeError("reference source attestation was not produced")
        raw_rows = list(raw_rows)
        reference_artifact = _fit_reference_artifact(
            raw_rows,
            checkpoint_path=args.checkpoint,
            preprocessing_checksum=preprocessing_checksum,
            scaler_checksum=scaler_checksum,
            framework_config=framework_config,
            framework_config_checksum=framework_config_checksum,
            source_attestation=source_attestation,
        )
        _validate_reference_artifact_contract(
            reference_artifact,
            framework_config=framework_config,
            framework_config_checksum=framework_config_checksum,
        )
        atomic_write_json(args.fit_reference_out, reference_artifact)
    if args.reference_in:
        reference_artifact = json.loads(args.reference_in.read_text(encoding="utf-8"))
        _validate_reference_artifact_contract(
            reference_artifact,
            framework_config=framework_config,
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
                framework_config,
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
        "inactive_no_score": 0,
        "raw_feature_scores": 0,
        "raw_sequence_scores": 0,
    }
    readiness_outcomes: dict[str, Counter[str]] = {
        "feature": Counter(),
        "sequence": Counter(),
    }

    def tracked_rows() -> Iterator[dict[str, Any]]:
        for row in rows:
            counters["samples"] += 1
            counters["scored"] += int(row["risk"] is not None)
            counters["inactive_no_score"] += int(
                row["risk"] is None and not bool(row.get("active", False))
            )
            counters["raw_feature_scores"] += int(row["feature_raw"] is not None)
            counters["raw_sequence_scores"] += int(row["sequence_raw"] is not None)
            trace = row.get("_readiness_trace", {})
            if isinstance(trace, Mapping):
                for branch in ("feature", "sequence"):
                    reasons = trace.get(branch, ())
                    if isinstance(reasons, (list, tuple)):
                        readiness_outcomes[branch].update(str(value) for value in reasons)
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
        "readiness_outcomes": {
            branch: dict(sorted(counts.items()))
            for branch, counts in readiness_outcomes.items()
        },
        "reference_build_diagnostics": (
            reference_artifact.get("build_diagnostics")
            if args.fit_reference_out and reference_artifact is not None
            else None
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
