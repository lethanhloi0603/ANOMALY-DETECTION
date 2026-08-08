from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli.score import (
    _fit_reference_artifact,
    _ready,
    _validate_reference_artifact_contract,
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
        "last_active_day": "2010-05-01",
        "sequence_days": 60,
        "transitions": 1500,
        "last_sequence_day": "2010-05-01",
    }
    support.update(changes)
    return {"support": support}


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


def test_legacy_v1_reference_is_not_silently_reinterpreted() -> None:
    with pytest.raises(ValueError, match="legacy framework.v4 artifact"):
        _validate_reference_artifact_contract(
            {
                "schema_version": "hierarchical-reference.v1",
                "fit_split": "TRAIN",
                "frozen_after_train": True,
            },
            framework_config_checksum="a" * 64,
        )


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
    )

    feature = artifact["branches"]["feature"]["GLOBAL"]["GLOBAL"]
    sequence = artifact["branches"]["sequence"]["GLOBAL"]["GLOBAL"]
    assert feature["support"]["feature_observation_days"] == 2
    assert feature["support"]["feature_observation_counts"] == [2] * 128
    assert sequence["stats"]["observation_count"] == 1
    assert sequence["support"]["sequence_days"] == 1
