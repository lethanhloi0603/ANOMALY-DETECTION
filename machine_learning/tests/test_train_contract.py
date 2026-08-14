from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cli import prepare_cert, train
from insider_ml.cert_data import SOURCE_FILES
from insider_ml.contracts import TRAIN_START

_COMMON_SOURCE_ARGUMENTS = {
    "preprocessing_checksum": "a" * 64,
    "scaler_checksum": "b" * 64,
    "sample_count": 17,
    "feature_schema_version": "feature128.v5",
    "sequence_schema_version": "sequence7.v4",
}
_FRAMEWORK_CONFIG = {
    "schema_version": "framework.v5",
    "training_sampling": {
        "endpoint_policy": "WEEKLY_TRAIN",
        "reference_fit_uses_all_train_endpoints": True,
    },
}
_FRAMEWORK_CHECKSUM = "f" * 64


def test_prepare_cert_defaults_to_explicit_smoke_user_cap(tmp_path: Path) -> None:
    args = prepare_cert.parse_args(
        [
            "--output",
            str(tmp_path / "train.npz"),
            "--split",
            "TRAIN",
            "--start-day",
            "2010-01-02",
            "--end-day",
            "2010-05-31",
        ]
    )

    assert args.max_users == 25


def test_npz_weekly_sampling_matches_locked_store_policy() -> None:
    user_id = "U001"
    stagger = int.from_bytes(
        hashlib.sha256(user_id.encode()).digest()[:4],
        "little",
    ) % 7
    matching_day = TRAIN_START + timedelta(days=stagger)
    windows = SimpleNamespace(
        sample_user_ids=[user_id, user_id],
        sample_end_days=[
            matching_day.isoformat(),
            (matching_day + timedelta(days=1)).isoformat(),
        ],
    )

    assert train._weekly_train_indices(windows) == [0]


def test_full_uncapped_npz_reuses_strict_source_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "train.npz"
    source.write_bytes(b"complete prepared Train source")
    expected_sha256 = train._sha256(source)
    observed: dict[str, Any] = {}

    def validate(path: Path, **kwargs: Any) -> dict[str, Any]:
        observed["path"] = path
        observed["kwargs"] = kwargs
        return {
            "schema_version": "reference-source-attestation.v1",
            "kind": "prepared_windows_npz",
            "complete_train": True,
            "source": str(source.resolve()),
            "source_sha256": expected_sha256,
            "manifest_sha256": "c" * 64,
            "manifest": str(source.with_suffix(".npz.manifest.json")),
            "start_day": "2010-01-02",
            "end_day": "2010-05-31",
            "endpoint_policy": "ALL",
            "max_users": None,
            "selected_user_count": 17,
            "samples": 17,
        }

    monkeypatch.setattr(train, "_validate_prepared_reference_source", validate)

    attestation = train._build_training_source_attestation(
        source,
        source_kind="prepared_windows_npz",
        max_samples=None,
        **_COMMON_SOURCE_ARGUMENTS,
    )

    assert observed == {
        "path": source,
        "kwargs": _COMMON_SOURCE_ARGUMENTS,
    }
    assert attestation["complete_train"] is True
    assert attestation["source_sha256"] == expected_sha256
    assert attestation["manifest_sha256"] == "c" * 64


def test_full_uncapped_store_reuses_strict_source_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "user-day.sqlite"
    source.write_bytes(b"complete store")
    calls: list[Path] = []

    def validate(path: Path) -> dict[str, Any]:
        calls.append(path)
        return {
            "schema_version": "reference-source-attestation.v1",
            "kind": "sqlite_user_day_store",
            "complete_train": True,
            "source": str(source.resolve()),
            "daily_materialization_end_day": "2010-05-31",
            "daily_materialization_signature": "d" * 64,
            "sources": {
                source_name: {
                    "fingerprint": {"path": f"{source_name}.csv"},
                    "max_rows": None,
                }
                for source_name in SOURCE_FILES
            },
        }

    monkeypatch.setattr(train, "_validate_store_reference_source", validate)

    attestation = train._build_training_source_attestation(
        source,
        source_kind="sqlite_user_day_store",
        max_samples=None,
        **_COMMON_SOURCE_ARGUMENTS,
    )

    assert calls == [source]
    assert attestation["complete_train"] is True
    assert attestation["source_sha256"] == train._sha256(source)
    assert attestation["daily_materialization_signature"] == "d" * 64


def test_capped_store_is_explicitly_non_production_without_strict_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "smoke.sqlite"
    source.write_bytes(b"capped store")

    def fail_if_called(_path: Path) -> dict[str, Any]:
        raise AssertionError("a capped store must not receive a complete Train attestation")

    monkeypatch.setattr(train, "_validate_store_reference_source", fail_if_called)

    attestation = train._build_training_source_attestation(
        source,
        source_kind="sqlite_user_day_store",
        max_samples=10,
        **_COMMON_SOURCE_ARGUMENTS,
    )

    assert attestation["complete_train"] is False
    assert attestation["reason_code"] == "TRAINING_SAMPLE_CAP_APPLIED"
    assert "smoke/research-only" in attestation["reason"]
    assert attestation["source_sha256"] == train._sha256(source)


def test_missing_or_short_npz_attestation_does_not_block_research_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "short-train.npz"
    source.write_bytes(b"short input")

    def reject(_path: Path, **_kwargs: Any) -> dict[str, Any]:
        raise ValueError("preparation manifest does not cover the locked Train range")

    monkeypatch.setattr(train, "_validate_prepared_reference_source", reject)

    attestation = train._build_training_source_attestation(
        source,
        source_kind="prepared_windows_npz",
        max_samples=None,
        **_COMMON_SOURCE_ARGUMENTS,
    )

    assert attestation["complete_train"] is False
    assert attestation["reason_code"] == "SOURCE_NOT_COMPLETE_TRAIN"
    assert "locked Train range" in attestation["reason"]
    assert attestation["source_sha256"] == train._sha256(source)


def test_user_capped_npz_checkpoint_is_explicitly_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "capped-train.npz"
    source.write_bytes(b"25-user prepared Train source")

    def reject(_path: Path, **_kwargs: Any) -> dict[str, Any]:
        raise ValueError("reference fitting refuses a user-capped NPZ source")

    monkeypatch.setattr(train, "_validate_prepared_reference_source", reject)
    attestation = train._build_training_source_attestation(
        source,
        source_kind="prepared_windows_npz",
        max_samples=None,
        **_COMMON_SOURCE_ARGUMENTS,
    )

    assert attestation["complete_train"] is False
    assert attestation["reason_code"] == "SOURCE_NOT_COMPLETE_TRAIN"
    assert "user-capped NPZ" in attestation["reason"]


def test_training_contract_is_train_only_and_resume_rejects_source_drift() -> None:
    attestation = {
        "schema_version": "reference-source-attestation.v1",
        "kind": "prepared_windows_npz",
        "complete_train": False,
        "source": "train.npz",
        "source_sha256": "e" * 64,
        "reason_code": "SOURCE_NOT_COMPLETE_TRAIN",
        "reason": "short source",
    }
    contract = train._training_contract_fields(
        attestation,
        framework_config=_FRAMEWORK_CONFIG,
        framework_config_checksum=_FRAMEWORK_CHECKSUM,
        training_endpoint_policy="WEEKLY_TRAIN",
    )

    assert contract == {
        "fit_split": "TRAIN",
        "frozen_after_train": True,
        "framework_schema_version": "framework.v5",
        "framework_config_sha256": _FRAMEWORK_CHECKSUM,
        "training_endpoint_policy": "WEEKLY_TRAIN",
        "training_source_attestation": attestation,
    }
    train._validate_resume_training_contract(contract, contract)

    changed_contract = {
        **contract,
        "training_source_attestation": {
            **attestation,
            "source_sha256": "a" * 64,
        },
    }
    with pytest.raises(SystemExit, match="training_source_attestation"):
        train._validate_resume_training_contract(contract, changed_contract)

    with pytest.raises(SystemExit, match="fit_split"):
        train._validate_resume_training_contract(
            {**contract, "fit_split": "TEST"},
            contract,
        )

    with pytest.raises(SystemExit, match="frozen_after_train"):
        train._validate_resume_training_contract(
            {**contract, "frozen_after_train": False},
            contract,
        )
