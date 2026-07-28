"""Evaluate immutable risk decisions against physically isolated labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from insider_ml.evaluation.metrics import (
    EvaluationInputError,
    evaluate,
    join_predictions_and_labels,
    load_labels_csv,
    load_labels_sqlite,
    load_predictions_csv,
    load_predictions_sqlite,
    load_universe_csv,
    threshold_for_alert_budget,
    threshold_for_validation_quantile,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute leakage-safe user-day metrics. Labels are read only by this "
            "evaluation process and are never written into the core database."
        )
    )
    predictions = parser.add_mutually_exclusive_group(required=True)
    predictions.add_argument("--core-sqlite", type=Path)
    predictions.add_argument("--predictions-csv", type=Path)
    labels = parser.add_mutually_exclusive_group(required=True)
    labels.add_argument("--labels-sqlite", type=Path)
    labels.add_argument("--labels-csv", type=Path)
    parser.add_argument(
        "--positive-only-answer-key",
        action="store_true",
        help=(
            "Explicitly confirm that each label row is positive and prediction "
            "user-days absent from the answer key are negatives."
        ),
    )
    parser.add_argument("--split", required=True, choices=["VALIDATION", "TEST"])
    parser.add_argument("--model-version")
    parser.add_argument("--config-version")
    parser.add_argument("--organization")
    parser.add_argument(
        "--universe-csv",
        type=Path,
        required=True,
        help="Independent eligible user-day manifest with user_id,day",
    )
    threshold = parser.add_mutually_exclusive_group(required=True)
    threshold.add_argument("--threshold", type=float)
    threshold.add_argument("--alert-budget-per-day", type=float)
    threshold.add_argument("--validation-quantile", type=float)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.core_sqlite and (
        not args.model_version or not args.config_version or not args.organization
    ):
        parser.error("--core-sqlite requires --organization, --model-version and --config-version")
    if (
        args.core_sqlite
        and args.labels_sqlite
        and args.core_sqlite.resolve() == args.labels_sqlite.resolve()
    ):
        parser.error("core and evaluation databases must be physically separate files")
    if not args.positive_only_answer_key:
        parser.error("label coverage is ambiguous; explicitly pass --positive-only-answer-key")
    if args.split == "TEST" and args.alert_budget_per_day is not None:
        parser.error("TEST cannot select a threshold; pass the threshold locked on VALIDATION")
    if args.split == "TEST" and args.validation_quantile is not None:
        parser.error("TEST cannot select a quantile; pass the threshold locked on VALIDATION")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be within [0, 1]")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.bootstrap < 0:
        parser.error("--bootstrap cannot be negative")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    universe = load_universe_csv(args.universe_csv, split=args.split)
    if args.predictions_csv:
        predictions = load_predictions_csv(args.predictions_csv, split=args.split)
    else:
        predictions = load_predictions_sqlite(
            args.core_sqlite,
            split=args.split,
            model_version=args.model_version,
            config_version=args.config_version,
            organization=args.organization,
        )
    if args.alert_budget_per_day is not None:
        prediction_rows = join_predictions_and_labels(predictions, [], universe)
        locked_threshold, budget = threshold_for_alert_budget(
            prediction_rows,
            args.alert_budget_per_day,
        )
        selection = {
            "method": "validation_alert_budget",
            "alerts_per_day": args.alert_budget_per_day,
            **budget,
        }
    elif args.validation_quantile is not None:
        prediction_rows = join_predictions_and_labels(predictions, [], universe)
        locked_threshold, quantile_details = threshold_for_validation_quantile(
            prediction_rows,
            args.validation_quantile,
        )
        selection = {
            "method": "validation_empirical_quantile",
            "labels_used": False,
            **quantile_details,
        }
    else:
        locked_threshold = args.threshold
        selection = {"method": "locked"}
    labels = (
        load_labels_csv(args.labels_csv, split=args.split)
        if args.labels_csv
        else load_labels_sqlite(args.labels_sqlite, split=args.split)
    )
    rows = join_predictions_and_labels(predictions, labels, universe)
    return evaluate(
        rows,
        split=args.split,
        threshold=locked_threshold,
        top_k=args.top_k,
        bootstrap_replicates=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        threshold_selection=selection,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = run(args)
        rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
            print(f"metrics written to {args.output}")
        else:
            print(rendered)
        return 0
    except (EvaluationInputError, OSError) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
