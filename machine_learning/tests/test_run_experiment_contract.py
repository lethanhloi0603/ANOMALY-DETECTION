from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cli import run_experiment


def _write_framework(path: Path, *, marker: int = 1) -> None:
    path.write_text(
        json.dumps({"schema_version": "framework.v5", "marker": marker}),
        encoding="utf-8",
    )


def test_resume_reference_requires_current_entry_contract_and_framework(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    framework = tmp_path / "framework.v5.json"
    _write_framework(framework)
    reference = tmp_path / "reference.json"
    artifact = {
        "schema_version": "hierarchical-reference.v2",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    reference.write_text(json.dumps(artifact), encoding="utf-8")
    calls: list[str] = []

    def validate(
        candidate: dict[str, object],
        *,
        framework_config: dict[str, object],
        framework_config_checksum: str,
    ) -> None:
        calls.append(framework_config_checksum)
        if candidate.get("entry_contract_version") != "hierarchical-reference-entry.v1":
            raise ValueError("missing deep entry contract")
        if candidate.get("framework_config_sha256") != framework_config_checksum:
            raise ValueError("framework drift")
        assert framework_config["schema_version"] == "framework.v5"

    monkeypatch.setattr(
        run_experiment,
        "_validate_reference_artifact_contract",
        validate,
    )
    assert not run_experiment._reference_current(
        reference,
        checkpoint=checkpoint,
        framework_config=framework,
    )

    artifact["entry_contract_version"] = "hierarchical-reference-entry.v1"
    artifact["framework_config_sha256"] = run_experiment._sha256(framework)
    reference.write_text(json.dumps(artifact), encoding="utf-8")
    assert run_experiment._reference_current(
        reference,
        checkpoint=checkpoint,
        framework_config=framework,
    )
    assert calls

    _write_framework(framework, marker=2)
    assert not run_experiment._reference_current(
        reference,
        checkpoint=checkpoint,
        framework_config=framework,
    )


def test_resume_score_manifest_is_invalidated_by_framework_drift(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    reference = tmp_path / "reference.json"
    reference.write_bytes(b"reference")
    output = tmp_path / "scores.csv"
    output.write_bytes(b"scores")
    framework = tmp_path / "framework.v5.json"
    _write_framework(framework)
    manifest = {
        "checkpoint_sha256": run_experiment._sha256(checkpoint),
        "reference_sha256": run_experiment._sha256(reference),
        "output_sha256": run_experiment._sha256(output),
        "framework_config_sha256": run_experiment._sha256(framework),
    }
    output.with_suffix(".csv.manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    assert run_experiment._score_current(
        output,
        checkpoint=checkpoint,
        reference=reference,
        framework_config=framework,
    )

    _write_framework(framework, marker=2)
    assert not run_experiment._score_current(
        output,
        checkpoint=checkpoint,
        reference=reference,
        framework_config=framework,
    )
