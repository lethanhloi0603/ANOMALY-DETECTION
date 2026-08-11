"""Run the complete leakage-safe CERT experiment through Test metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from cli.score import _load_framework_config, _validate_reference_artifact_contract
from insider_ml.artifacts import atomic_write_json
from insider_ml.contracts import TEST_END
from insider_ml.stream_store import (
    connect_store,
    metadata_get,
    store_readiness_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Full low-RAM experiment: streaming store -> Train -> frozen references -> "
            "Validation threshold -> frozen Test metrics."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=root / "data" / "raw" / "cert4.2")
    parser.add_argument(
        "--store",
        type=Path,
        default=root / "data" / "processed" / "cert4.2_user_days.sqlite",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=root / "data" / "artifacts" / "experiment_v1",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=root / "data" / "evaluation" / "experiment_v1",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--framework-config",
        type=Path,
        default=root / "backend" / "config" / "framework.v5.json",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-store-build",
        action="store_true",
        help="Require an already completed store instead of invoking the resumable builder",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def _score_current(
    output: Path,
    *,
    checkpoint: Path,
    reference: Path | None,
    framework_config: Path,
) -> bool:
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if not output.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    return (
        manifest.get("checkpoint_sha256") == _sha256(checkpoint)
        and manifest.get("framework_config_sha256") == _sha256(framework_config)
        and manifest.get("reference_sha256")
        == (_sha256(reference) if reference is not None else None)
        and manifest.get("output_sha256") == _sha256(output)
    )


def _score_manifest(output: Path) -> dict[str, Any]:
    return json.loads(
        output.with_suffix(output.suffix + ".manifest.json").read_text(encoding="utf-8")
    )


def _reference_current(
    reference: Path,
    *,
    checkpoint: Path,
    framework_config: Path,
) -> bool:
    if not reference.is_file():
        return False
    try:
        artifact = json.loads(reference.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(artifact, dict):
        return False
    if artifact.get("checkpoint_sha256") != _sha256(checkpoint):
        return False
    try:
        loaded_framework = _load_framework_config(framework_config)
        _validate_reference_artifact_contract(
            artifact,
            framework_config=loaded_framework,
            framework_config_checksum=_sha256(framework_config),
        )
    except (OSError, ValueError):
        return False
    return True


def _validate_full_store(path: Path) -> tuple[dict[str, int], dict[str, object]]:
    connection = connect_store(path, read_only=True)
    try:
        if metadata_get(connection, "daily_materialization_complete") != "true":
            raise ValueError("daily store materialization is incomplete")
        if metadata_get(connection, "daily_materialization_end_day") != TEST_END.isoformat():
            raise ValueError(
                f"full experiment requires store through locked Test end {TEST_END}"
            )
        for source in ("LOGON", "DEVICE", "FILE", "HTTP", "EMAIL"):
            raw = metadata_get(connection, f"source_complete_{source}")
            if raw is None or json.loads(raw).get("max_rows") is not None:
                raise ValueError(
                    f"full experiment refuses capped/incomplete source {source}"
                )
        counts = {
            str(split): int(count)
            for split, count in connection.execute(
                """
                SELECT split,COUNT(*)
                FROM daily_tensors
                WHERE split IN ('TRAIN','VALIDATION','TEST')
                GROUP BY split
                """
            )
        }
        readiness = store_readiness_report(connection)
    finally:
        connection.close()
    if set(counts) != {"TRAIN", "VALIDATION", "TEST"} or any(
        value == 0 for value in counts.values()
    ):
        raise ValueError(f"store split coverage is incomplete: {counts}")
    if not bool(readiness["at_least_one_branch_ready"]):
        raise ValueError(
            f"no global branch can calibrate Validation/Test: {readiness}"
        )
    return counts, readiness


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs < 1 or args.batch_size < 1 or args.bootstrap < 0:
        raise ValueError("epochs/batch-size must be positive and bootstrap non-negative")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.evaluation_dir.mkdir(parents=True, exist_ok=True)
    _load_framework_config(args.framework_config)
    python = sys.executable
    common = [python, "-m"]

    if not args.skip_store_build:
        _run(
            [
                *common,
                "cli.build_store",
                "--raw-root",
                str(args.raw_root),
                "--store",
                str(args.store),
            ]
        )
    if not args.store.is_file():
        raise FileNotFoundError(f"completed daily store not found: {args.store}")
    store_split_counts, readiness_preflight = _validate_full_store(args.store)

    checkpoint = args.artifact_dir / "tcn_transformer_ae.v4.pt"
    train_command = [
        *common,
        "cli.train",
        "--store",
        str(args.store),
        "--output",
        str(checkpoint),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--endpoint-policy",
        "weekly_train",
        "--framework-config",
        str(args.framework_config),
    ]
    if args.resume:
        train_command.append("--resume")
    _run(train_command)

    train_scores = args.artifact_dir / "train_raw_scores.csv"
    references = args.artifact_dir / "references.train.json"
    if not (
        args.resume
        and references.is_file()
        and _score_current(
            train_scores,
            checkpoint=checkpoint,
            reference=None,
            framework_config=args.framework_config,
        )
        and _reference_current(
            references,
            checkpoint=checkpoint,
            framework_config=args.framework_config,
        )
    ):
        _run(
            [
                *common,
                "cli.score",
                "--store",
                str(args.store),
                "--split",
                "TRAIN",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(train_scores),
                "--fit-reference-out",
                str(references),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
                "--framework-config",
                str(args.framework_config),
            ]
        )

    validation_predictions = args.artifact_dir / "validation_predictions.csv"
    if not (
        args.resume
        and _score_current(
            validation_predictions,
            checkpoint=checkpoint,
            reference=references,
            framework_config=args.framework_config,
        )
    ):
        _run(
            [
                *common,
                "cli.score",
                "--store",
                str(args.store),
                "--split",
                "VALIDATION",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(validation_predictions),
                "--reference-in",
                str(references),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
                "--framework-config",
                str(args.framework_config),
            ]
        )
    validation_score_manifest = _score_manifest(validation_predictions)
    if int(validation_score_manifest["scored"]) == 0:
        raise RuntimeError(
            "Validation produced zero calibrated scores; inspect frozen reference support"
        )

    validation_universe = args.evaluation_dir / "validation_universe.csv"
    validation_labels = args.evaluation_dir / "validation_positive_labels.csv"
    _run(
        [
            *common,
            "cli.prepare_evaluation",
            "--raw-root",
            str(args.raw_root),
            "--split",
            "VALIDATION",
            "--universe-out",
            str(validation_universe),
            "--labels-out",
            str(validation_labels),
        ]
    )
    validation_metrics = args.artifact_dir / "validation.metrics.json"
    _run(
        [
            *common,
            "cli.evaluate_metrics",
            "--predictions-csv",
            str(validation_predictions),
            "--labels-csv",
            str(validation_labels),
            "--positive-only-answer-key",
            "--universe-csv",
            str(validation_universe),
            "--split",
            "VALIDATION",
            "--validation-quantile",
            "0.995",
            "--top-k",
            "10",
            "--bootstrap",
            str(args.bootstrap),
            "--bootstrap-seed",
            "20260728",
            "--output",
            str(validation_metrics),
        ]
    )
    validation_report = json.loads(validation_metrics.read_text(encoding="utf-8"))
    locked_threshold = float(validation_report["threshold"])

    test_predictions = args.artifact_dir / "test_predictions.csv"
    if not (
        args.resume
        and _score_current(
            test_predictions,
            checkpoint=checkpoint,
            reference=references,
            framework_config=args.framework_config,
        )
    ):
        _run(
            [
                *common,
                "cli.score",
                "--store",
                str(args.store),
                "--split",
                "TEST",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(test_predictions),
                "--reference-in",
                str(references),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
                "--framework-config",
                str(args.framework_config),
            ]
        )
    test_universe = args.evaluation_dir / "test_universe.csv"
    test_labels = args.evaluation_dir / "test_positive_labels.csv"
    _run(
        [
            *common,
            "cli.prepare_evaluation",
            "--raw-root",
            str(args.raw_root),
            "--split",
            "TEST",
            "--universe-out",
            str(test_universe),
            "--labels-out",
            str(test_labels),
        ]
    )
    test_metrics = args.artifact_dir / "test.metrics.json"
    _run(
        [
            *common,
            "cli.evaluate_metrics",
            "--predictions-csv",
            str(test_predictions),
            "--labels-csv",
            str(test_labels),
            "--positive-only-answer-key",
            "--universe-csv",
            str(test_universe),
            "--split",
            "TEST",
            "--threshold",
            str(locked_threshold),
            "--top-k",
            "10",
            "--bootstrap",
            str(args.bootstrap),
            "--bootstrap-seed",
            "20260728",
            "--output",
            str(test_metrics),
        ]
    )
    test_report = json.loads(test_metrics.read_text(encoding="utf-8"))
    report = {
        "schema_version": "cert-experiment-run.v1",
        "status": "COMPLETE",
        "store": str(args.store.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "framework_config_sha256": _sha256(args.framework_config),
        "references": str(references.resolve()),
        "validation_metrics": str(validation_metrics.resolve()),
        "test_metrics": str(test_metrics.resolve()),
        "locked_validation_threshold": locked_threshold,
        "store_split_counts": store_split_counts,
        "readiness_preflight": readiness_preflight,
        "training_endpoint_policy": "WEEKLY_TRAIN",
        "validation": {
            "ranking": validation_report["ranking"],
            "coverage": validation_report["coverage"],
        },
        "test": {
            "ranking": test_report["ranking"],
            "coverage": test_report["coverage"],
            "at_threshold": test_report["at_threshold"],
            "operations": test_report["operations"],
        },
    }
    report_path = args.artifact_dir / "experiment_report.json"
    atomic_write_json(report_path, report)
    report["report"] = str(report_path.resolve())
    return report


def main() -> int:
    try:
        result = run(parse_args())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"experiment failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
