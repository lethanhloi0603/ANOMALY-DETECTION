from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one or more A0-A5 score files with validation-fitted thresholds."
    )
    parser.add_argument(
        "--scores",
        action="append",
        required=True,
        help="Experiment in NAME=path.csv form. Repeat for A0-A5 comparisons.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--quantiles", default="0.95,0.99,0.995,0.999")
    return parser.parse_args()


def parse_score_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected NAME=path.csv, got: {spec}")
    name, path = spec.split("=", 1)
    return name.strip(), Path(path.strip())


def incident_statistics(test: pd.DataFrame) -> tuple[int, int, list[int]]:
    """Return incident count, detected count, and delays for detected incidents.

    CERT scenario/user pairs are treated as incidents. Day-level recall remains
    available separately, while this metric answers the operational question:
    did the detector alert at least once during the incident?
    """
    delays: list[int] = []
    positives = test[test["label_day"] == 1].copy()
    if positives.empty:
        return 0, 0, delays
    positives["date"] = pd.to_datetime(positives["date"], errors="coerce")
    group_columns = ["user"] + (["scenario"] if "scenario" in positives.columns else [])
    total_incidents = 0
    for _, incident in positives.groupby(group_columns, dropna=False):
        total_incidents += 1
        first_positive = incident["date"].min()
        detected = incident[incident["prediction"] == 1]
        if not detected.empty:
            delays.append(int((detected["date"].min() - first_positive).days))
    return total_incidents, len(delays), delays


def normalize_scenario(value) -> str:
    """Keep scenario identifiers stable when CSV inference turns `1` into `1.0`."""
    if pd.isna(value):
        return "unknown"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def evaluate_one(name: str, path: Path, quantiles: list[float]):
    frame = pd.read_csv(path, low_memory=False)
    required = {"split", "score", "label_day", "user", "date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    validation = frame[frame["split"] == "validation"]
    test = frame[frame["split"] == "test"].copy()
    if validation.empty or test.empty:
        raise ValueError(f"{path} must contain non-empty validation and test splits")

    metrics = []
    scenario_rows = []
    for quantile in quantiles:
        threshold = float(np.quantile(validation["score"].astype(float), quantile))
        test["prediction"] = (test["score"].astype(float) > threshold).astype(int)
        labels = test["label_day"].astype(int).to_numpy()
        predictions = test["prediction"].to_numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
        auprc = float(average_precision_score(labels, test["score"].astype(float))) if labels.sum() > 0 else np.nan
        positives = labels == 1
        negatives = ~positives
        true_positives = int((positives & (predictions == 1)).sum())
        false_positives = int((negatives & (predictions == 1)).sum())
        true_negatives = int((negatives & (predictions == 0)).sum())
        false_negatives = int((positives & (predictions == 0)).sum())
        predicted_positives = int((predictions == 1).sum())
        positive_days = int(positives.sum())
        negative_days = int(negatives.sum())
        prevalence = positive_days / max(len(test), 1)
        total_incidents, detected_incidents, delays = incident_statistics(test)
        metrics.append({
            "experiment": name,
            "quantile": quantile,
            "threshold": threshold,
            "test_user_days": int(len(test)),
            "positive_user_days": positive_days,
            "negative_user_days": negative_days,
            "prevalence": prevalence,
            "auprc": auprc,
            "auprc_random_baseline": prevalence,
            "auprc_lift_over_random": auprc / prevalence if prevalence > 0 else np.nan,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "predicted_positive_user_days": predicted_positives,
            "false_positives_per_user_day": false_positives / max(len(test), 1),
            "false_positive_rate": false_positives / max(negative_days, 1),
            "specificity": true_negatives / max(negative_days, 1),
            "alerts_per_1000_user_days": 1000.0 * predicted_positives / max(len(test), 1),
            "false_alerts_per_1000_benign_user_days": 1000.0 * false_positives / max(negative_days, 1),
            "mean_detection_delay_days": float(np.mean(delays)) if delays else np.nan,
            "median_detection_delay_days": float(np.median(delays)) if delays else np.nan,
            "total_incidents": total_incidents,
            "detected_incidents": detected_incidents,
            "incident_recall": detected_incidents / max(total_incidents, 1),
        })
        if "scenario" in test.columns:
            positive_frame = test[test["label_day"] == 1]
            for scenario, group in positive_frame.groupby("scenario", dropna=False):
                scenario_incidents = int(group["user"].nunique())
                detected_scenario_incidents = int(group.loc[group["prediction"] == 1, "user"].nunique())
                scenario_rows.append({
                    "experiment": name,
                    "quantile": quantile,
                    "scenario": normalize_scenario(scenario),
                    "positive_user_days": int(len(group)),
                    "scenario_recall": float(group["prediction"].mean()) if len(group) else np.nan,
                    "total_incidents": scenario_incidents,
                    "detected_incidents": detected_scenario_incidents,
                    "scenario_incident_recall": detected_scenario_incidents / max(scenario_incidents, 1),
                })
    return metrics, scenario_rows


def main():
    args = parse_args()
    quantiles = [float(value) for value in args.quantiles.split(",") if value.strip()]
    if any(not 0 < value < 1 for value in quantiles):
        raise ValueError("Every quantile must be between 0 and 1")
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    all_scenarios = []
    for spec in args.scores:
        name, path = parse_score_spec(spec)
        metrics, scenarios = evaluate_one(name, path, quantiles)
        all_metrics.extend(metrics)
        all_scenarios.extend(scenarios)
    metrics_frame = pd.DataFrame(all_metrics)
    scenario_frame = pd.DataFrame(all_scenarios)
    metrics_frame.to_csv(output_dir / "evaluation_metrics.csv", index=False, float_format="%.10g")
    scenario_frame.to_csv(output_dir / "scenario_recall.csv", index=False, float_format="%.10g")
    summary = {
        "experiments": sorted(metrics_frame["experiment"].unique().tolist()),
        "quantiles": quantiles,
        "metrics_file": str(output_dir / "evaluation_metrics.csv"),
        "scenario_file": str(output_dir / "scenario_recall.csv"),
        "warning": None if metrics_frame["positive_user_days"].max() > 0 else "No positive test user-days; AUPRC and detection delay are undefined.",
    }
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
