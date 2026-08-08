from __future__ import annotations

import csv
import json
from pathlib import Path

from insider_ml.artifacts import atomic_write_csv, atomic_write_json


def test_atomic_json_replaces_existing_artifact(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    output.write_text('{"status":"old"}\n', encoding="utf-8")

    atomic_write_json(output, {"status": "complete", "rows": 2})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "rows": 2,
        "status": "complete",
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_csv_writes_complete_rows(tmp_path: Path) -> None:
    output = tmp_path / "scores.csv"
    atomic_write_csv(
        output,
        fieldnames=("user_id", "risk"),
        rows=(
            {"user_id": "U1", "risk": 0.9},
            {"user_id": "U2", "risk": 0.1},
        ),
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"user_id": "U1", "risk": "0.9"},
        {"user_id": "U2", "risk": "0.1"},
    ]
    assert not list(tmp_path.glob("*.tmp"))
