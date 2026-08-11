from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import pytest

from cli.score import (
    _calibrate_row,
    _fit_reference_artifact,
    _ready,
    _reference_entry_checksum,
    _validate_checkpoint_training_contract,
    _validate_prepared_reference_source,
    _validate_reference_artifact_contract,
    _validate_store_reference_source,
    run,
)
from insider_ml.cert_data import SOURCE_FILES
from insider_ml.stream_store import (
    connect_store,
    initialize_store,
    metadata_set,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "backend" / "config" / "framework.v5.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _person_entry(**changes: object) -> dict[str, object]:
    support: dict[str, object] = {
        "active_days": 60,
        "span_days": 90,
        "mean_feature_coverage": 0.90,
        "minimum_nonzero_feature_support": 40,
        "last_active_day": "2010-05-02",
        "sequence_days": 60,
        "transitions": 1500,
        "last_sequence_day": "2010-05-02",
    }
    support.update(changes)
    return {"support": support}


def _feature_role_entry(**changes: object) -> dict[str, object]:
    user_ids = ["U001", *(f"U{index:03d}" for index in range(2, 17))]
    support: dict[str, object] = {
        "users": 16,
        "user_ids": user_ids,
        "observations": 301,
        "observations_by_user": {"U001": 1},
        "recent_30d_observations": 101,
        "recent_30d_observations_by_user": {"U001": 1},
        "mean_feature_coverage": 0.90,
        "minimum_nonzero_feature_support": 200,
    }
    support.update(changes)
    return {"support": support}


def _feature_global_entry(**changes: object) -> dict[str, object]:
    support: dict[str, object] = {
        "users": 200,
        "observations": 10_000,
        "mean_feature_coverage": 0.90,
    }
    support.update(changes)
    return {"support": support}


def _reference_contract(
    framework_config: dict[str, object],
    *,
    checksum: str,
    checkpoint_path: Path,
) -> dict[str, object]:
    return _fit_reference_artifact(
        _role_ready_reference_rows(200),
        checkpoint_path=checkpoint_path,
        preprocessing_checksum="b" * 64,
        scaler_checksum="c" * 64,
        framework_config=deepcopy(framework_config),
        framework_config_checksum=checksum,
        source_attestation=_source_attestation(),
    )


def _source_attestation() -> dict[str, object]:
    return {
        "schema_version": "reference-source-attestation.v1",
        "kind": "prepared_windows_npz",
        "complete_train": True,
        "source": "train.npz",
        "source_sha256": "e" * 64,
        "manifest": "train.npz.manifest.json",
        "manifest_sha256": "f" * 64,
        "start_day": "2010-01-02",
        "end_day": "2010-05-31",
        "endpoint_policy": "ALL",
        "max_users": None,
        "selected_user_count": 1,
        "samples": 1,
    }


def _reference_row(
    *,
    user_id: str,
    day: str,
    role: str,
    feature_raw: float,
) -> dict[str, object]:
    return {
        "user_id": user_id,
        "day": day,
        "split": "TRAIN",
        "role": role,
        "role_epoch": f"{user_id}|{role}|2010-01-02",
        "feature_raw": feature_raw,
        "sequence_raw": feature_raw + 0.5,
        "sequence_length": 35,
        "active": True,
        "feature_coverage": 1.0,
        "_feature_mask": bytes([0xFF] * 16),
    }


def _role_ready_reference_rows(personal_count: int) -> list[dict[str, object]]:
    rows = [
        _reference_row(
            user_id="U001",
            day=(
                date(2010, 2, 2)
                + timedelta(days=(89 * index // max(personal_count - 1, 1)))
            ).isoformat(),
            role="Engineer",
            feature_raw=float(index),
        )
        for index in range(personal_count)
    ]
    for user_number in range(2, 17):
        for day_offset in range(20):
            rows.append(
                _reference_row(
                    user_id=f"U{user_number:03d}",
                    day=f"2010-05-{day_offset + 12:02d}",
                    role="Engineer",
                    feature_raw=float(1_000 + user_number * 20 + day_offset),
                )
            )
    return rows


def _global_ready_role_not_ready_rows(
    personal_count: int,
) -> list[dict[str, object]]:
    rows = _role_ready_reference_rows(personal_count)[:personal_count]
    for user_number in range(2, 201):
        role = f"Role-{user_number:03d}"
        for observation in range(50):
            rows.append(
                _reference_row(
                    user_id=f"U{user_number:03d}",
                    day=f"2010-05-{observation % 30 + 1:02d}",
                    role=role,
                    feature_raw=float(1_000 + user_number * 50 + observation),
                )
            )
    return rows


def _fit_feature_reference(
    tmp_path: Path,
    *,
    personal_count: int,
    global_parent: bool = False,
) -> dict[str, object]:
    checkpoint = tmp_path / (
        f"checkpoint-{personal_count}-{'global' if global_parent else 'role'}.pt"
    )
    checkpoint.write_bytes(b"immutable-checkpoint")
    rows = (
        _global_ready_role_not_ready_rows(personal_count)
        if global_parent
        else _role_ready_reference_rows(personal_count)
    )
    return _fit_reference_artifact(
        rows,
        checkpoint_path=checkpoint,
        preprocessing_checksum="b" * 64,
        scaler_checksum="c" * 64,
        framework_config=_config(),
        framework_config_checksum="d" * 64,
        source_attestation=_source_attestation(),
    )


def _personal_feature_entry(artifact: dict[str, object]) -> dict[str, object]:
    return artifact["branches"]["feature"]["PERSON"][
        "U001|Engineer|2010-01-02"
    ]


def _reseal(entry: dict[str, object]) -> None:
    entry["entry_checksum_sha256"] = _reference_entry_checksum(entry)


def _write_store_attestation(
    path: Path,
    *,
    end_day: str = "2010-05-31",
    omitted_source: str | None = None,
    capped_source: str | None = None,
) -> None:
    connection = connect_store(path)
    initialize_store(connection)
    metadata_set(connection, "daily_materialization_complete", "true")
    metadata_set(connection, "daily_materialization_end_day", end_day)
    metadata_set(connection, "daily_materialization_signature", "a" * 64)
    for source in SOURCE_FILES:
        if source == omitted_source:
            continue
        metadata_set(
            connection,
            f"source_complete_{source}",
            {
                "fingerprint": {
                    "path": f"{source.lower()}.csv",
                    "size": 1,
                    "mtime_ns": 1,
                },
                "max_rows": 100 if source == capped_source else None,
            },
        )
    connection.commit()
    connection.close()


def _write_preparation_manifest(
    input_path: Path,
    **changes: object,
) -> Path:
    payload: dict[str, object] = {
        "schema_version": "cert-preparation-manifest.v1",
        "split": "TRAIN",
        "start_day": "2010-01-02",
        "end_day": "2010-05-31",
        "smoke_row_cap_per_source": None,
        "users": ["U001", "U002"],
        "selected_user_count": 2,
        "max_users": None,
        "output": str(input_path.resolve()),
        "output_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "preprocessing_checksum": "b" * 64,
        "scaler_checksum": "c" * 64,
        "samples": 2,
        "feature_schema_version": "feature128.v5",
        "sequence_schema_version": "sequence7.v4",
        "endpoint_policy": "ALL",
    }
    payload.update(changes)
    manifest_path = input_path.with_suffix(input_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def test_validation_test_person_readiness_is_frozen_at_train_end() -> None:
    config = _config()
    for split, day in (
        ("VALIDATION", "2010-09-30"),
        ("TEST", "2011-05-17"),
    ):
        assert _ready(
            branch="feature",
            level="PERSON",
            entry=_person_entry(),
            user_id="U001",
            sequence_length=2,
            day=day,
            split=split,
            role="Engineer",
            framework_config=config,
        )


def test_frozen_person_staleness_is_evaluated_at_first_day_after_train() -> None:
    config = _config()
    for branch, last_day_key in (
        ("feature", "last_active_day"),
        ("sequence", "last_sequence_day"),
    ):
        for split in ("VALIDATION", "TEST"):
            assert _ready(
                branch=branch,
                level="PERSON",
                entry=_person_entry(**{last_day_key: "2010-05-02"}),
                user_id="U001",
                sequence_length=2,
                day="2011-05-17" if split == "TEST" else "2010-06-01",
                split=split,
                role="Engineer",
                framework_config=config,
            )
            assert not _ready(
                branch=branch,
                level="PERSON",
                entry=_person_entry(**{last_day_key: "2010-05-01"}),
                user_id="U001",
                sequence_length=2,
                day="2011-05-17" if split == "TEST" else "2010-06-01",
                split=split,
                role="Engineer",
                framework_config=config,
            )


def test_live_readiness_uses_score_day_for_staleness() -> None:
    assert not _ready(
        branch="feature",
        level="PERSON",
        entry=_person_entry(),
        user_id="U001",
        sequence_length=2,
        day="2011-05-17",
        split="PRODUCTION",
        role="Engineer",
        framework_config=_config(),
    )


def test_feature_person_v5_boundary_is_enforced() -> None:
    config = _config()
    assert _ready(
        branch="feature",
        level="PERSON",
        entry=_person_entry(),
        user_id="U001",
        sequence_length=2,
        day="2010-06-01",
        split="VALIDATION",
        role="Engineer",
        framework_config=config,
    )
    assert not _ready(
        branch="feature",
        level="PERSON",
        entry=_person_entry(active_days=59),
        user_id="U001",
        sequence_length=2,
        day="2010-06-01",
        split="VALIDATION",
        role="Engineer",
        framework_config=config,
    )


def test_sequence_person_v5_boundary_is_enforced() -> None:
    config = _config()
    assert _ready(
        branch="sequence",
        level="PERSON",
        entry=_person_entry(),
        user_id="U001",
        sequence_length=2,
        day="2010-06-01",
        split="VALIDATION",
        role="Engineer",
        framework_config=config,
    )
    assert not _ready(
        branch="sequence",
        level="PERSON",
        entry=_person_entry(transitions=1499),
        user_id="U001",
        sequence_length=2,
        day="2010-06-01",
        split="VALIDATION",
        role="Engineer",
        framework_config=config,
    )


def test_feature_role_requires_coverage_and_per_feature_support_independently() -> None:
    config = _config()
    common = {
        "branch": "feature",
        "level": "ROLE",
        "user_id": "U001",
        "sequence_length": 2,
        "day": "2010-06-01",
        "split": "VALIDATION",
        "role": "Engineer",
        "framework_config": config,
    }
    assert _ready(entry=_feature_role_entry(), **common)
    assert not _ready(
        entry=_feature_role_entry(mean_feature_coverage=0.899999),
        **common,
    )
    assert not _ready(
        entry=_feature_role_entry(minimum_nonzero_feature_support=199),
        **common,
    )


def test_feature_global_uses_mean_feature_coverage() -> None:
    config = _config()
    common = {
        "branch": "feature",
        "level": "GLOBAL",
        "user_id": "U001",
        "sequence_length": 2,
        "day": "2010-06-01",
        "split": "VALIDATION",
        "role": "Engineer",
        "framework_config": config,
    }
    assert _ready(entry=_feature_global_entry(), **common)
    assert not _ready(
        entry=_feature_global_entry(mean_feature_coverage=0.899999),
        **common,
    )


def test_legacy_v1_reference_is_not_silently_reinterpreted() -> None:
    config = _config()
    with pytest.raises(ValueError, match="legacy framework.v4 artifact"):
        _validate_reference_artifact_contract(
            {
                "schema_version": "hierarchical-reference.v1",
                "fit_split": "TRAIN",
                "frozen_after_train": True,
            },
            framework_config=config,
            framework_config_checksum="a" * 64,
        )


def test_reference_embedded_framework_config_must_match_external_config(
    tmp_path: Path,
) -> None:
    config = _config()
    checksum = "a" * 64
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"immutable-checkpoint")
    artifact = _reference_contract(
        config,
        checksum=checksum,
        checkpoint_path=checkpoint,
    )
    _validate_reference_artifact_contract(
        artifact,
        framework_config=config,
        framework_config_checksum=checksum,
    )

    mutated = deepcopy(artifact)
    mutated["framework_config"]["readiness"]["feature"]["person"][
        "min_active_days"
    ] = 59
    with pytest.raises(ValueError, match="embedded reference framework config"):
        _validate_reference_artifact_contract(
            mutated,
            framework_config=config,
            framework_config_checksum=checksum,
        )


@pytest.mark.parametrize(
    ("personal_count", "expected_method"),
    [
        (60, "parent_shrunk_ecdf.v1"),
        (199, "parent_shrunk_ecdf.v1"),
        (200, "empirical_cdf.v1"),
    ],
)
def test_fitted_personal_entry_contract_switches_at_200(
    tmp_path: Path,
    personal_count: int,
    expected_method: str,
) -> None:
    artifact = _fit_feature_reference(
        tmp_path,
        personal_count=personal_count,
    )
    entry = _personal_feature_entry(artifact)

    assert set(entry) == {
        "entry_schema_version",
        "identity",
        "stats",
        "support",
        "calibrator",
        "entry_checksum_sha256",
    }
    assert entry["entry_schema_version"] == "hierarchical-reference-entry.v1"
    assert entry["identity"] == {
        "branch": "feature",
        "level": "PERSON",
        "scope_key": "U001|Engineer|2010-01-02",
        "subject_user_id": "U001",
        "role": "Engineer",
    }
    assert entry["stats"]["observation_count"] == personal_count
    assert len(entry["stats"]["sorted_scores"]) == personal_count
    assert entry["calibrator"]["method"] == expected_method
    assert entry["entry_checksum_sha256"] == _reference_entry_checksum(entry)

    if personal_count < 200:
        role_entry = artifact["branches"]["feature"]["ROLE"]["Engineer"]
        assert entry["calibrator"] == {
            "method": "parent_shrunk_ecdf.v1",
            "standalone_target_observations": 200,
            "parent": {
                "branch": "feature",
                "level": "ROLE",
                "scope_key": "Engineer",
                "entry_checksum_sha256": role_entry["entry_checksum_sha256"],
            },
        }
    else:
        assert entry["calibrator"] == {"method": "empirical_cdf.v1"}


def test_personal_calibrator_pins_global_when_role_is_not_ready(
    tmp_path: Path,
) -> None:
    artifact = _fit_feature_reference(
        tmp_path,
        personal_count=60,
        global_parent=True,
    )
    entry = _personal_feature_entry(artifact)
    global_entry = artifact["branches"]["feature"]["GLOBAL"]["GLOBAL"]

    assert entry["calibrator"]["parent"] == {
        "branch": "feature",
        "level": "GLOBAL",
        "scope_key": "GLOBAL",
        "entry_checksum_sha256": global_entry["entry_checksum_sha256"],
    }
    _validate_reference_artifact_contract(
        artifact,
        framework_config=_config(),
        framework_config_checksum="d" * 64,
    )


def test_small_personal_reference_is_omitted_without_a_ready_parent(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-no-parent.pt"
    checkpoint.write_bytes(b"immutable-checkpoint")
    artifact = _fit_reference_artifact(
        _role_ready_reference_rows(60)[:60],
        checkpoint_path=checkpoint,
        preprocessing_checksum="b" * 64,
        scaler_checksum="c" * 64,
        framework_config=_config(),
        framework_config_checksum="d" * 64,
        source_attestation=_source_attestation(),
    )

    person_scope = "U001|Engineer|2010-01-02"
    assert person_scope not in artifact["branches"]["feature"]["PERSON"]
    assert person_scope not in artifact["branches"]["sequence"]["PERSON"]
    _validate_reference_artifact_contract(
        artifact,
        framework_config=_config(),
        framework_config_checksum="d" * 64,
    )


def test_personal_calibration_mixes_personal_and_pinned_parent_cdfs(
    tmp_path: Path,
) -> None:
    artifact = _fit_feature_reference(tmp_path, personal_count=60)
    person_entry = _personal_feature_entry(artifact)
    role_entry = artifact["branches"]["feature"]["ROLE"]["Engineer"]
    raw_score = 30.0

    calibrated = _calibrate_row(
        {
            "user_id": "U001",
            "day": "2010-06-01",
            "split": "VALIDATION",
            "role": "Engineer",
            "role_epoch": "U001|Engineer|2010-01-02",
            "feature_raw": raw_score,
            "sequence_raw": None,
            "sequence_length": 2,
        },
        artifact,
        _config(),
    )

    personal_scores = person_entry["stats"]["sorted_scores"]
    parent_scores = role_entry["stats"]["sorted_scores"]
    personal_cdf = sum(value <= raw_score for value in personal_scores) / len(
        personal_scores
    )
    parent_cdf = sum(value <= raw_score for value in parent_scores) / len(
        parent_scores
    )
    expected = 0.30 * personal_cdf + 0.70 * parent_cdf
    assert calibrated["feature_level"] == "PERSON"
    assert calibrated["feature_calibrated"] == pytest.approx(expected)
    assert calibrated["risk"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "mutation",
    [
        "tamper",
        "count_mismatch",
        "unsorted_scores",
        "identity_path_mismatch",
        "expected_parent_mismatch",
    ],
)
def test_deep_reference_validator_rejects_invalid_entry_contract(
    tmp_path: Path,
    mutation: str,
) -> None:
    artifact = _fit_feature_reference(tmp_path, personal_count=60)
    config = _config()
    _validate_reference_artifact_contract(
        artifact,
        framework_config=config,
        framework_config_checksum="d" * 64,
    )
    entry = _personal_feature_entry(artifact)

    if mutation == "tamper":
        entry["stats"]["location"] += 0.25
    elif mutation == "count_mismatch":
        entry["stats"]["observation_count"] += 1
        _reseal(entry)
    elif mutation == "unsorted_scores":
        scores = entry["stats"]["sorted_scores"]
        scores[0], scores[1] = scores[1], scores[0]
        _reseal(entry)
    elif mutation == "identity_path_mismatch":
        entry["identity"]["scope_key"] = "U999|Engineer|2010-01-02"
        _reseal(entry)
    else:
        global_entry = artifact["branches"]["feature"]["GLOBAL"]["GLOBAL"]
        entry["calibrator"]["parent"] = {
            "branch": "feature",
            "level": "GLOBAL",
            "scope_key": "GLOBAL",
            "entry_checksum_sha256": global_entry["entry_checksum_sha256"],
        }
        _reseal(entry)

    with pytest.raises(ValueError):
        _validate_reference_artifact_contract(
            artifact,
            framework_config=config,
            framework_config_checksum="d" * 64,
        )


def test_reference_validator_rejects_non_finite_sorted_score(
    tmp_path: Path,
) -> None:
    artifact = _fit_feature_reference(tmp_path, personal_count=60)
    config = _config()
    entry = _personal_feature_entry(artifact)
    entry["stats"]["sorted_scores"][0] = float("nan")

    with pytest.raises(ValueError, match="canonical finite JSON"):
        _reseal(entry)
    with pytest.raises(ValueError, match="finite numbers"):
        _validate_reference_artifact_contract(
            artifact,
            framework_config=config,
            framework_config_checksum="d" * 64,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_consumed_field",
        "wrong_ratio_type",
        "date_outside_train",
        "per_user_total_mismatch",
        "feature_counter_shape",
        "span_too_large",
        "recent_transitions_exceed_total",
        "unexpected_last_active_day",
        "mean_coverage_mismatch",
    ],
)
def test_reference_validator_rejects_invalid_support_contract(
    tmp_path: Path,
    mutation: str,
) -> None:
    artifact = _fit_feature_reference(tmp_path, personal_count=60)
    entry = _personal_feature_entry(artifact)
    support = entry["support"]
    if mutation == "missing_consumed_field":
        support.pop("active_days")
    elif mutation == "wrong_ratio_type":
        support["mean_feature_coverage"] = "1.0"
    elif mutation == "date_outside_train":
        support["last_active_day"] = "2010-06-01"
    elif mutation == "per_user_total_mismatch":
        support["transitions_by_user"]["U001"] += 1
    elif mutation == "feature_counter_shape":
        support["feature_observation_counts"] = support[
            "feature_observation_counts"
        ][:-1]
    elif mutation == "span_too_large":
        support["span_days"] = 151
    elif mutation == "recent_transitions_exceed_total":
        support["recent_30d_transitions_by_user"]["U001"] = (
            support["transitions_by_user"]["U001"] + 1
        )
        support["recent_30d_transitions"] = support[
            "recent_30d_transitions_by_user"
        ]["U001"]
    elif mutation == "unexpected_last_active_day":
        support["active_days"] = 0
    else:
        support["mean_feature_coverage"] = 0.5
    _reseal(entry)

    with pytest.raises(ValueError, match="support"):
        _validate_reference_artifact_contract(
            artifact,
            framework_config=_config(),
            framework_config_checksum="d" * 64,
        )


def test_frozen_reference_cannot_be_applied_to_train_rows() -> None:
    config = _config()
    artifact = {"branches": {"feature": {}, "sequence": {}}}
    row = {
        "user_id": "U001",
        "day": "2010-05-31",
        "split": "TRAIN",
        "role": "Engineer",
        "role_epoch": "U001|Engineer|2010-01-02",
        "feature_raw": None,
        "sequence_raw": None,
        "sequence_length": 0,
    }
    with pytest.raises(ValueError, match="only be applied to rows after Train end"):
        _calibrate_row(row, artifact, config)

    contradictory = {**row, "split": "VALIDATION"}
    with pytest.raises(ValueError, match="only be applied to rows after Train end"):
        _calibrate_row(contradictory, artifact, config)


def test_reference_fit_rejects_smoke_sample_cap() -> None:
    args = argparse.Namespace(
        batch_size=1,
        store=Path("unused.sqlite"),
        split="TRAIN",
        input=None,
        fit_reference_out=Path("reference.json"),
        max_samples=10,
    )
    with pytest.raises(ValueError, match="complete Train split"):
        run(args)


def test_reference_calibration_requires_complete_train_checkpoint() -> None:
    config = _config()
    config_checksum = "d" * 64
    complete = {
        "fit_split": "TRAIN",
        "frozen_after_train": True,
        "framework_schema_version": "framework.v5",
        "framework_config_sha256": config_checksum,
        "training_endpoint_policy": "WEEKLY_TRAIN",
        "feature_schema_version": "feature128.v5",
        "sequence_schema_version": "sequence7.v4",
        "feature_value_space": "robust_scaled",
        "preprocessing_checksum": "a" * 64,
        "scaler_checksum": "b" * 64,
        "training_samples": 10,
        "training_source_attestation": _source_attestation(),
    }
    _validate_checkpoint_training_contract(
        complete,
        framework_config=config,
        framework_config_checksum=config_checksum,
    )

    for invalid in (
        {**complete, "fit_split": "TEST"},
        {**complete, "frozen_after_train": False},
        {**complete, "training_endpoint_policy": "ALL"},
        {**complete, "preprocessing_checksum": None},
        {
            **complete,
            "training_source_attestation": {
                **complete["training_source_attestation"],
                "complete_train": False,
            },
        },
        {
            **complete,
            "training_source_attestation": {
                **complete["training_source_attestation"],
                "max_users": 25,
            },
        },
    ):
        with pytest.raises(ValueError, match="complete Train|frozen Train|framework.v5"):
            _validate_checkpoint_training_contract(
                invalid,
                framework_config=config,
                framework_config_checksum=config_checksum,
            )


def test_store_reference_source_requires_full_uncapped_train_attestation(
    tmp_path: Path,
) -> None:
    complete_store = tmp_path / "complete.sqlite"
    _write_store_attestation(complete_store)
    attestation = _validate_store_reference_source(complete_store)
    assert attestation["complete_train"] is True
    assert set(attestation["sources"]) == set(SOURCE_FILES)

    short_store = tmp_path / "short.sqlite"
    _write_store_attestation(short_store, end_day="2010-05-30")
    with pytest.raises(ValueError, match="locked Train end"):
        _validate_store_reference_source(short_store)

    capped_store = tmp_path / "capped.sqlite"
    _write_store_attestation(capped_store, capped_source="EMAIL")
    with pytest.raises(ValueError, match="refuses capped source EMAIL"):
        _validate_store_reference_source(capped_store)

    incomplete_store = tmp_path / "incomplete.sqlite"
    _write_store_attestation(incomplete_store, omitted_source="HTTP")
    with pytest.raises(ValueError, match="completion metadata for HTTP"):
        _validate_store_reference_source(incomplete_store)


def test_npz_reference_source_requires_matching_full_train_manifest(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "train.npz"
    input_path.write_bytes(b"prepared-windows")
    manifest_path = _write_preparation_manifest(input_path)
    arguments = {
        "preprocessing_checksum": "b" * 64,
        "scaler_checksum": "c" * 64,
        "sample_count": 2,
        "feature_schema_version": "feature128.v5",
        "sequence_schema_version": "sequence7.v4",
    }
    attestation = _validate_prepared_reference_source(input_path, **arguments)
    assert attestation["complete_train"] is True
    assert attestation["source_sha256"] == hashlib.sha256(
        input_path.read_bytes()
    ).hexdigest()

    _write_preparation_manifest(input_path, smoke_row_cap_per_source=100)
    with pytest.raises(ValueError, match="refuses a capped NPZ"):
        _validate_prepared_reference_source(input_path, **arguments)

    _write_preparation_manifest(input_path, max_users=25)
    with pytest.raises(ValueError, match="refuses a user-capped NPZ"):
        _validate_prepared_reference_source(input_path, **arguments)

    _write_preparation_manifest(input_path, endpoint_policy="WEEKLY_TRAIN")
    with pytest.raises(ValueError, match="all locked Train endpoints"):
        _validate_prepared_reference_source(input_path, **arguments)

    _write_preparation_manifest(input_path, start_day="2010-01-03")
    with pytest.raises(ValueError, match="exact locked Train range"):
        _validate_prepared_reference_source(input_path, **arguments)

    _write_preparation_manifest(input_path, output_sha256="0" * 64)
    with pytest.raises(ValueError, match="output checksum"):
        _validate_prepared_reference_source(input_path, **arguments)

    manifest_path.unlink()
    with pytest.raises(ValueError, match="preparation manifest beside the input"):
        _validate_prepared_reference_source(input_path, **arguments)


def test_reference_artifact_keeps_feature_denominator_and_filters_short_sequences(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"immutable-checkpoint")
    packed_mask = bytes([0xFF] * 16)
    rows = [
        {
            "user_id": "U001",
            "day": "2010-05-01",
            "split": "TRAIN",
            "role": "Engineer",
            "role_epoch": "U001|Engineer|2010-01-02",
            "feature_raw": 0.1,
            "sequence_raw": 0.1,
            "sequence_length": 1,
            "active": True,
            "feature_coverage": 1.0,
            "_feature_mask": packed_mask,
        },
        {
            "user_id": "U001",
            "day": "2010-05-02",
            "split": "TRAIN",
            "role": "Engineer",
            "role_epoch": "U001|Engineer|2010-01-02",
            "feature_raw": 0.2,
            "sequence_raw": 0.2,
            "sequence_length": 2,
            "active": True,
            "feature_coverage": 1.0,
            "_feature_mask": packed_mask,
        },
    ]

    artifact = _fit_reference_artifact(
        rows,
        checkpoint_path=checkpoint,
        preprocessing_checksum="b" * 64,
        scaler_checksum="c" * 64,
        framework_config=_config(),
        framework_config_checksum="d" * 64,
        source_attestation=_source_attestation(),
    )

    feature = artifact["branches"]["feature"]["GLOBAL"]["GLOBAL"]
    sequence = artifact["branches"]["sequence"]["GLOBAL"]["GLOBAL"]
    assert feature["support"]["feature_observation_days"] == 2
    assert feature["support"]["feature_observation_counts"] == [2] * 128
    assert sequence["stats"]["observation_count"] == 1
    assert sequence["support"]["sequence_days"] == 1
