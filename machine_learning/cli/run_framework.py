"""One-command bounded smoke run from CERT raw CSV through inference."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

from insider_ml.artifacts import atomic_write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Execute a bounded real-CERT smoke run: prepare -> train -> "
            "raw score -> frozen-reference calibration check."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=root / "data" / "raw" / "cert4.2")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=root / "data" / "processed" / "smoke",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=root / "data" / "artifacts" / "smoke",
    )
    parser.add_argument("--end-day", type=date.fromisoformat, default=date(2010, 1, 3))
    parser.add_argument("--max-rows-per-source", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args(argv)


def _run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    store = args.processed_dir / "user_days.sqlite"
    checkpoint = args.artifact_dir / "low_ram_tcn_transformer_ae.v4.pt"
    raw_scores = args.artifact_dir / "low_ram_train_raw_scores.csv"
    references = args.artifact_dir / "low_ram_references.train.json"
    predictions = args.artifact_dir / "low_ram_train_predictions.csv"

    common = [sys.executable, "-m"]
    _run(
        [
            *common,
            "cli.build_store",
            "--raw-root",
            str(args.raw_root),
            "--store",
            str(store),
            "--end-day",
            args.end_day.isoformat(),
            "--max-rows-per-source",
            str(args.max_rows_per_source),
        ]
    )
    _run(
        [
            *common,
            "cli.train",
            "--store",
            str(store),
            "--output",
            str(checkpoint),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--max-samples",
            "8",
            "--endpoint-policy",
            "weekly_train",
            "--device",
            "cpu",
        ]
    )
    _run(
        [
            *common,
            "cli.score",
            "--store",
            str(store),
            "--split",
            "TRAIN",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(raw_scores),
            "--fit-reference-out",
            str(references),
            "--batch-size",
            str(args.batch_size),
            "--max-samples",
            "20",
            "--device",
            "cpu",
        ]
    )
    _run(
        [
            *common,
            "cli.score",
            "--store",
            str(store),
            "--split",
            "TRAIN",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(predictions),
            "--reference-in",
            str(references),
            "--batch-size",
            str(args.batch_size),
            "--max-samples",
            "20",
            "--device",
            "cpu",
        ]
    )
    with predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    report = {
        "schema_version": "framework-smoke-run.v1",
        "status": "PASS",
        "scope": "bounded real-CERT execution, not a research metric result",
        "samples": len(rows),
        "raw_feature_scores": sum(bool(row["feature_raw"]) for row in rows),
        "raw_sequence_scores": sum(bool(row["sequence_raw"]) for row in rows),
        "calibrated_scores": sum(bool(row["risk"]) for row in rows),
        "expected_smoke_no_score_reason": (
            "The two-day shard intentionally cannot satisfy locked Person/Role/Global "
            "readiness support."
        ),
        "store": str(store.resolve()),
        "store_size_bytes": store.stat().st_size,
        "checkpoint": str(checkpoint.resolve()),
        "reference": str(references.resolve()),
        "predictions": str(predictions.resolve()),
    }
    report_path = args.artifact_dir / "smoke_report.json"
    atomic_write_json(report_path, report)
    report["report"] = str(report_path.resolve())
    return report


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
