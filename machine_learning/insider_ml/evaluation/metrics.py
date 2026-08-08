"""Leakage-safe metrics for immutable user-day risk decisions.

Labels are joined only inside this evaluation package. A missing risk decision
is evaluated as ``NO_SCORE`` with effective risk zero so coverage failures
cannot silently improve reported metrics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean, median
from typing import Any

SPLIT_RANGES = {
    "TRAIN": (date(2010, 1, 2), date(2010, 5, 31)),
    "VALIDATION": (date(2010, 6, 1), date(2010, 9, 30)),
    "TEST": (date(2010, 10, 1), date(2011, 5, 17)),
}
REFERENCE_LEVELS = ("PERSON", "ROLE", "GLOBAL", "NO_SCORE")


class EvaluationInputError(ValueError):
    """Raised when evaluation inputs violate their locked contract."""


@dataclass(frozen=True, slots=True)
class Prediction:
    user_id: str
    day: date
    risk: float | None
    feature_level: str = "NO_SCORE"
    sequence_level: str = "NO_SCORE"
    status: str = "NO_SCORE"
    threshold: float | None = None
    is_alert: bool | None = None

    def __post_init__(self) -> None:
        if not self.user_id.strip():
            raise EvaluationInputError("prediction user_id cannot be empty")
        if self.risk is not None and (not math.isfinite(self.risk) or not 0 <= self.risk <= 1):
            raise EvaluationInputError(f"risk must be finite and within [0, 1], got {self.risk}")
        if self.threshold is not None and (
            not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1
        ):
            raise EvaluationInputError(
                f"stored threshold must be finite and within [0, 1], got {self.threshold}"
            )

    @property
    def key(self) -> tuple[str, date]:
        return self.user_id, self.day

    @property
    def effective_risk(self) -> float:
        return 0.0 if self.risk is None else self.risk


@dataclass(frozen=True, slots=True)
class PositiveLabel:
    user_id: str
    day: date
    incident_id: str | None = None
    scenario: str | None = None

    def __post_init__(self) -> None:
        if not self.user_id.strip():
            raise EvaluationInputError("label user_id cannot be empty")

    @property
    def key(self) -> tuple[str, date]:
        return self.user_id, self.day


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    user_id: str
    day: date
    risk: float
    positive: bool
    feature_level: str
    sequence_level: str
    no_score: bool
    missing_prediction: bool
    incident_id: str | None = None
    scenario: str | None = None
    stored_threshold: float | None = None
    stored_is_alert: bool | None = None


def _parse_day(value: str, *, field: str = "day") -> date:
    try:
        return date.fromisoformat(value.strip())
    except (AttributeError, ValueError) as exc:
        raise EvaluationInputError(f"{field} must be an ISO date, got {value!r}") from exc


def _parse_risk(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        risk = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationInputError(f"risk must be numeric or blank, got {value!r}") from exc
    if not math.isfinite(risk) or not 0 <= risk <= 1:
        raise EvaluationInputError(f"risk must be finite and within [0, 1], got {risk}")
    return risk


def _normalize_level(value: object) -> str:
    normalized = str(value or "NO_SCORE").strip().upper()
    return normalized if normalized in REFERENCE_LEVELS else "NO_SCORE"


def _parse_optional_bool(value: object) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise EvaluationInputError(f"boolean value expected, got {value!r}")


def _in_split(day: date, split: str) -> bool:
    start, end = SPLIT_RANGES[split]
    return start <= day <= end


def _require_columns(
    fieldnames: Sequence[str] | None,
    required: set[str],
    *,
    source: Path,
) -> None:
    available = set(fieldnames or ())
    missing = sorted(required - available)
    if missing:
        raise EvaluationInputError(f"{source} is missing required columns: {missing}")


def load_predictions_csv(path: Path, *, split: str) -> list[Prediction]:
    """Load immutable user-day predictions from a CSV export."""

    rows: list[Prediction] = []
    seen: set[tuple[str, date]] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, {"user_id", "day", "risk"}, source=path)
        for line_number, raw in enumerate(reader, start=2):
            user_id = str(raw["user_id"]).strip()
            day = _parse_day(raw["day"])
            if not user_id:
                raise EvaluationInputError(f"{path}:{line_number}: user_id is empty")
            if not _in_split(day, split):
                continue
            key = (user_id, day)
            if key in seen:
                raise EvaluationInputError(f"{path}:{line_number}: duplicate user-day {key}")
            seen.add(key)
            risk = _parse_risk(raw.get("risk"))
            status = str(raw.get("status") or ("SCORED" if risk is not None else "NO_SCORE"))
            rows.append(
                Prediction(
                    user_id=user_id,
                    day=day,
                    risk=risk,
                    feature_level=_normalize_level(raw.get("feature_level")),
                    sequence_level=_normalize_level(raw.get("sequence_level")),
                    status=status.strip().upper(),
                    threshold=_parse_risk(raw.get("threshold")),
                    is_alert=_parse_optional_bool(raw.get("is_alert")),
                )
            )
    return rows


def load_universe_csv(path: Path, *, split: str) -> list[tuple[str, date]]:
    """Load the independently materialized eligible user-day universe."""

    keys: list[tuple[str, date]] = []
    seen: set[tuple[str, date]] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, {"user_id", "day"}, source=path)
        for line_number, raw in enumerate(reader, start=2):
            user_id = str(raw["user_id"]).strip()
            day = _parse_day(raw["day"])
            if not user_id:
                raise EvaluationInputError(f"{path}:{line_number}: user_id is empty")
            if not _in_split(day, split):
                continue
            key = (user_id, day)
            if key in seen:
                raise EvaluationInputError(f"{path}:{line_number}: duplicate universe key {key}")
            seen.add(key)
            keys.append(key)
    if not keys:
        raise EvaluationInputError(f"eligible user-day universe is empty for split={split}")
    return keys


def load_predictions_sqlite(
    path: Path,
    *,
    split: str,
    model_version: str,
    config_version: str,
    organization: str,
) -> list[Prediction]:
    """Read core decisions without ever attaching the evaluation database."""

    if not path.is_file():
        raise EvaluationInputError(f"core SQLite database does not exist: {path}")
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only=ON")
        query = """
            SELECT
                u.external_user_id,
                ra.day,
                ra.risk,
                ra.status,
                COALESCE(fs.selected_level, 'no_score') AS feature_level,
                COALESCE(ss.selected_level, 'no_score') AS sequence_level,
                ra.threshold,
                ra.is_alert
            FROM risk_assessments AS ra
            JOIN users AS u ON u.id = ra.user_id
            JOIN organizations AS o ON o.id = ra.organization_id
            LEFT JOIN branch_scores AS fs ON fs.id = ra.feature_score_id
            LEFT JOIN branch_scores AS ss ON ss.id = ra.sequence_score_id
            WHERE o.slug = ?
              AND ra.split = ?
              AND ra.model_version = ?
              AND ra.config_version = ?
            ORDER BY ra.day, u.external_user_id
        """
        raw_rows = connection.execute(
            query,
            (organization, split.lower(), model_version, config_version),
        ).fetchall()
    except sqlite3.Error as exc:
        raise EvaluationInputError(f"cannot read core decisions from {path}: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()

    predictions = [
        Prediction(
            user_id=str(row[0]),
            day=_parse_day(str(row[1])),
            risk=_parse_risk(row[2]),
            status=str(row[3]).upper(),
            feature_level=_normalize_level(row[4]),
            sequence_level=_normalize_level(row[5]),
            threshold=_parse_risk(row[6]),
            is_alert=bool(row[7]) if row[7] is not None else None,
        )
        for row in raw_rows
    ]
    if not predictions:
        raise EvaluationInputError(
            "no risk assessments match "
            f"organization={organization!r}, split={split}, "
            f"model_version={model_version!r}, "
            f"config_version={config_version!r}"
        )
    return predictions


def load_labels_csv(path: Path, *, split: str) -> list[PositiveLabel]:
    """Load expanded positive user-days; absent rows remain negatives."""

    labels: list[PositiveLabel] = []
    seen: set[tuple[str, date]] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, {"user_id", "day"}, source=path)
        for line_number, raw in enumerate(reader, start=2):
            if "is_positive" in raw:
                marker = _parse_optional_bool(raw.get("is_positive"))
                if marker is not True:
                    raise EvaluationInputError(
                        f"{path}:{line_number}: is_positive must be true in "
                        "positive-only answer-key mode"
                    )
            user_id = str(raw["user_id"]).strip()
            day = _parse_day(raw["day"])
            if not user_id:
                raise EvaluationInputError(f"{path}:{line_number}: user_id is empty")
            if not _in_split(day, split):
                continue
            key = (user_id, day)
            if key in seen:
                raise EvaluationInputError(f"{path}:{line_number}: duplicate label {key}")
            seen.add(key)
            labels.append(
                PositiveLabel(
                    user_id=user_id,
                    day=day,
                    incident_id=str(raw.get("incident_id") or "").strip() or None,
                    scenario=str(raw.get("scenario") or "").strip() or None,
                )
            )
    return labels


def load_labels_sqlite(path: Path, *, split: str) -> list[PositiveLabel]:
    """Load labels from the physically isolated evaluation database."""

    if not path.is_file():
        raise EvaluationInputError(f"evaluation SQLite database does not exist: {path}")
    start, end = SPLIT_RANGES[split]
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only=ON")
        raw_rows = connection.execute(
            """
            SELECT user_id, day, incident_id, scenario
            FROM evaluation_labels
            WHERE day BETWEEN ? AND ?
            ORDER BY day, user_id
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    except sqlite3.Error as exc:
        raise EvaluationInputError(f"cannot read evaluation labels from {path}: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()
    return [
        PositiveLabel(
            user_id=str(row[0]),
            day=_parse_day(str(row[1])),
            incident_id=str(row[2]) if row[2] is not None else None,
            scenario=str(row[3]) if row[3] is not None else None,
        )
        for row in raw_rows
    ]


def join_predictions_and_labels(
    predictions: Iterable[Prediction],
    labels: Iterable[PositiveLabel],
    universe: Iterable[tuple[str, date]] | None = None,
) -> list[EvaluationRow]:
    prediction_map: dict[tuple[str, date], Prediction] = {}
    for prediction in predictions:
        if prediction.key in prediction_map:
            raise EvaluationInputError(f"duplicate prediction {prediction.key}")
        prediction_map[prediction.key] = prediction

    label_map: dict[tuple[str, date], PositiveLabel] = {}
    for label in labels:
        if label.key in label_map:
            raise EvaluationInputError(f"duplicate label {label.key}")
        label_map[label.key] = label

    if universe is None:
        keys_set = set(prediction_map) | set(label_map)
    else:
        keys_set = set(universe)
        if not keys_set:
            raise EvaluationInputError("eligible user-day universe is empty")
        prediction_outside = sorted(set(prediction_map) - keys_set)
        label_outside = sorted(set(label_map) - keys_set)
        if prediction_outside or label_outside:
            raise EvaluationInputError(
                "prediction/label keys fall outside the eligible user-day universe: "
                f"predictions={prediction_outside[:5]}, labels={label_outside[:5]}"
            )
    keys = sorted(keys_set, key=lambda item: (item[1], item[0]))
    rows: list[EvaluationRow] = []
    for user_id, day in keys:
        prediction = prediction_map.get((user_id, day))
        label = label_map.get((user_id, day))
        missing = prediction is None
        rows.append(
            EvaluationRow(
                user_id=user_id,
                day=day,
                risk=prediction.effective_risk if prediction else 0.0,
                positive=label is not None,
                feature_level=prediction.feature_level if prediction else "NO_SCORE",
                sequence_level=prediction.sequence_level if prediction else "NO_SCORE",
                no_score=prediction is None or prediction.risk is None,
                missing_prediction=missing,
                incident_id=label.incident_id if label else None,
                scenario=label.scenario if label else None,
                stored_threshold=prediction.threshold if prediction else None,
                stored_is_alert=prediction.is_alert if prediction else None,
            )
        )
    if not rows:
        raise EvaluationInputError("evaluation universe is empty")
    return rows


def average_precision(rows: Sequence[EvaluationRow]) -> float | None:
    positives = sum(row.positive for row in rows)
    if positives == 0:
        return None
    ordered = sorted(rows, key=lambda row: (-row.risk, row.day, row.user_id))
    true_positives = 0
    seen = 0
    previous_recall = 0.0
    area = 0.0
    index = 0
    while index < len(ordered):
        score = ordered[index].risk
        end = index
        group_positives = 0
        while end < len(ordered) and ordered[end].risk == score:
            group_positives += int(ordered[end].positive)
            end += 1
        true_positives += group_positives
        seen += end - index
        recall = true_positives / positives
        precision = true_positives / seen
        area += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return area


def roc_auc(rows: Sequence[EvaluationRow]) -> float | None:
    positive_count = sum(row.positive for row in rows)
    negative_count = len(rows) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ordered = sorted(rows, key=lambda row: row.risk)
    negatives_before = 0
    concordant = 0.0
    index = 0
    while index < len(ordered):
        score = ordered[index].risk
        end = index
        group_positives = 0
        group_negatives = 0
        while end < len(ordered) and ordered[end].risk == score:
            if ordered[end].positive:
                group_positives += 1
            else:
                group_negatives += 1
            end += 1
        concordant += group_positives * negatives_before
        concordant += 0.5 * group_positives * group_negatives
        negatives_before += group_negatives
        index = end
    return concordant / (positive_count * negative_count)


def threshold_metrics(rows: Sequence[EvaluationRow], threshold: float) -> dict[str, Any]:
    true_positive = false_positive = true_negative = false_negative = 0
    for row in rows:
        alerted = not row.no_score and row.risk >= threshold
        if alerted and row.positive:
            true_positive += 1
        elif alerted:
            false_positive += 1
        elif row.positive:
            false_negative += 1
        else:
            true_negative += 1
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "alerts": true_positive + false_positive,
    }


def threshold_for_alert_budget(
    rows: Sequence[EvaluationRow],
    alerts_per_day: float,
) -> tuple[float, dict[str, Any]]:
    if not math.isfinite(alerts_per_day) or alerts_per_day <= 0:
        raise EvaluationInputError("alerts_per_day must be a positive finite number")
    prediction_rows = [row for row in rows if not row.missing_prediction]
    scored_rows = [row for row in prediction_rows if not row.no_score]
    if not scored_rows:
        raise EvaluationInputError("cannot select an alert budget without scored predictions")
    day_count = len({row.day for row in prediction_rows})
    budget = max(1, math.floor(alerts_per_day * day_count))
    scores = sorted((row.risk for row in scored_rows), reverse=True)
    if budget >= len(scores):
        return 0.0, {
            "target_alerts": len(scores),
            "achieved_alerts": len(scores),
            "unused_budget": 0,
            "tie_exclusion": False,
        }
    cutoff = scores[budget - 1]
    next_score = scores[budget]
    tie_exclusion = cutoff == next_score
    threshold = math.nextafter(cutoff, math.inf) if tie_exclusion else (cutoff + next_score) / 2
    achieved_alerts = sum(score >= threshold for score in scores)
    return threshold, {
        "target_alerts": budget,
        "achieved_alerts": achieved_alerts,
        "unused_budget": budget - achieved_alerts,
        "cutoff_score": cutoff,
        "next_score": next_score,
        "tie_exclusion": tie_exclusion,
    }


def threshold_for_validation_quantile(
    rows: Sequence[EvaluationRow],
    quantile: float,
) -> tuple[float, dict[str, Any]]:
    """Lock a label-free threshold from the empirical Validation risk distribution."""

    if not math.isfinite(quantile) or not 0 < quantile < 1:
        raise EvaluationInputError("validation quantile must be within (0, 1)")
    scored_rows = [row for row in rows if not row.missing_prediction and not row.no_score]
    if not scored_rows:
        raise EvaluationInputError("cannot select a quantile without scored predictions")
    scores = sorted(row.risk for row in scored_rows)
    index = max(0, math.ceil(quantile * len(scores)) - 1)
    threshold = scores[index]
    achieved_alerts = sum(score >= threshold for score in scores)
    return threshold, {
        "quantile": quantile,
        "sample_count": len(scores),
        "nearest_rank": index + 1,
        "achieved_alerts": achieved_alerts,
        "achieved_tail_rate": achieved_alerts / len(scores),
        "tie_overflow": achieved_alerts > len(scores) - index,
    }


def recall_at_top_k_per_day(rows: Sequence[EvaluationRow], top_k: int) -> dict[str, Any]:
    if top_k < 1:
        raise EvaluationInputError("top_k must be positive")
    by_day: dict[date, list[EvaluationRow]] = defaultdict(list)
    for row in rows:
        by_day[row.day].append(row)
    selected: set[tuple[str, date]] = set()
    for day_rows in by_day.values():
        ordered = sorted(
            (row for row in day_rows if not row.no_score),
            key=lambda row: (-row.risk, row.user_id),
        )
        selected.update((row.user_id, row.day) for row in ordered[:top_k])
    positive_count = sum(row.positive for row in rows)
    detected = sum(row.positive and (row.user_id, row.day) in selected for row in rows)
    return {
        "value": detected / positive_count if positive_count else None,
        "detected_positive_user_days": detected,
        "positive_user_days": positive_count,
        "selected_user_days": len(selected),
        "top_k_per_day": top_k,
    }


def incident_metrics(rows: Sequence[EvaluationRow], threshold: float) -> dict[str, Any]:
    incidents: dict[str, list[EvaluationRow]] = defaultdict(list)
    for row in rows:
        if row.positive and row.incident_id:
            incidents[row.incident_id].append(row)
    delays: list[int] = []
    detected = 0
    for incident_rows in incidents.values():
        start = min(row.day for row in incident_rows)
        alerted_days = [
            row.day for row in incident_rows if not row.no_score and row.risk >= threshold
        ]
        if alerted_days:
            detected += 1
            delays.append((min(alerted_days) - start).days)
    return {
        "incident_detection_rate": detected / len(incidents) if incidents else None,
        "detected_incidents": detected,
        "incident_count": len(incidents),
        "mean_time_to_detect_days": mean(delays) if delays else None,
        "median_time_to_detect_days": median(delays) if delays else None,
    }


def scenario_recall(rows: Sequence[EvaluationRow], threshold: float) -> dict[str, Any]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if not row.positive or not row.scenario:
            continue
        counts[row.scenario][1] += 1
        if not row.no_score and row.risk >= threshold:
            counts[row.scenario][0] += 1
    return {
        scenario: {
            "recall": detected / total,
            "detected_positive_user_days": detected,
            "positive_user_days": total,
        }
        for scenario, (detected, total) in sorted(counts.items())
    }


def fallback_rates(rows: Sequence[EvaluationRow], field: str) -> dict[str, Any]:
    counts = {level: 0 for level in REFERENCE_LEVELS}
    for row in rows:
        level = getattr(row, field)
        counts[level if level in counts else "NO_SCORE"] += 1
    total = len(rows)
    return {level: {"rate": count / total, "count": count} for level, count in counts.items()}


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def user_block_confidence_intervals(
    rows: Sequence[EvaluationRow],
    threshold: float,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        return {}
    blocks: dict[str, list[EvaluationRow]] = defaultdict(list)
    for row in rows:
        blocks[row.user_id].append(row)
    users = sorted(blocks)
    if not users:
        return {}
    generator = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sampled: list[EvaluationRow] = []
        for user_id in generator.choices(users, k=len(users)):
            sampled.extend(blocks[user_id])
        ranking = {
            "auprc": average_precision(sampled),
            "roc_auc": roc_auc(sampled),
        }
        point = threshold_metrics(sampled, threshold)
        for name, value in {**ranking, **point}.items():
            if name in {"precision", "recall", "f1", "auprc", "roc_auc"} and value is not None:
                samples[name].append(float(value))
    return {
        name: {
            "low": _percentile(values, 0.025),
            "high": _percentile(values, 0.975),
            "requested_replicates": replicates,
            "valid_replicates": len(values),
            "block": "user",
        }
        for name, values in sorted(samples.items())
    }


def _manifest_checksum(
    rows: Sequence[EvaluationRow],
    *,
    split: str,
    threshold: float,
) -> str:
    payload = {
        "split": split,
        "threshold": threshold,
        "rows": [
            {
                "user_id": row.user_id,
                "day": row.day.isoformat(),
                "risk": row.risk,
                "positive": row.positive,
                "incident_id": row.incident_id,
                "scenario": row.scenario,
                "feature_level": row.feature_level,
                "sequence_level": row.sequence_level,
                "no_score": row.no_score,
                "missing_prediction": row.missing_prediction,
                "stored_threshold": row.stored_threshold,
                "stored_is_alert": row.stored_is_alert,
            }
            for row in rows
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evaluate(
    rows: Sequence[EvaluationRow],
    *,
    split: str,
    threshold: float,
    top_k: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    threshold_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the document's mandatory metric set."""

    if split not in {"VALIDATION", "TEST"}:
        raise EvaluationInputError("metrics may only be selected on VALIDATION or run on TEST")
    if not math.isfinite(threshold):
        raise EvaluationInputError("threshold must be finite")
    if split == "TEST":
        for row in rows:
            if row.missing_prediction:
                continue
            expected_alert = not row.no_score and row.risk >= threshold
            if row.stored_threshold is not None and not math.isclose(
                row.stored_threshold,
                threshold,
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise EvaluationInputError(
                    "TEST threshold differs from an immutable stored decision: "
                    f"{row.user_id}/{row.day}"
                )
            if row.stored_is_alert is not None and row.stored_is_alert != expected_alert:
                raise EvaluationInputError(
                    "TEST alert flag is inconsistent with its locked risk/threshold: "
                    f"{row.user_id}/{row.day}"
                )
    point = threshold_metrics(rows, threshold)
    day_count = len({row.day for row in rows})
    positive_count = sum(row.positive for row in rows)
    false_alerts_per_1000 = point["false_positive"] / len(rows) * 1000
    alerts_per_day = point["alerts"] / day_count
    warnings: list[str] = []
    if positive_count == 0:
        warnings.append("No positive labels are present; recall and AUPRC are undefined.")
    if positive_count == len(rows):
        warnings.append("No negative user-days are present; ROC-AUC is undefined.")
    missing_predictions = sum(row.missing_prediction for row in rows)
    if missing_predictions:
        warnings.append(
            f"{missing_predictions} eligible user-day rows had no prediction and were "
            "penalized as NO_SCORE with risk=0."
        )
    result = {
        "schema_version": "evaluation.metrics.v1",
        "split": split,
        "label_contract": {
            "mode": "positive_only_answer_key",
            "positive_definition": "A (user_id, day) row present in the answer key.",
            "negative_definition": (
                "An eligible-universe (user_id, day) absent from the answer key."
            ),
        },
        "metric_contract": {
            "average_precision": "Tie-grouped user-day average precision.",
            "recall_at_top_k_per_day": (
                "Micro recall over positive user-days; rank independently per day "
                "with user_id as deterministic tie-break."
            ),
            "time_to_detect": (
                "Days from first labeled positive incident day to first alerted "
                "labeled day; undetected incidents excluded from delay aggregation."
            ),
            "confidence_interval": "Seeded user-block bootstrap.",
        },
        "threshold": threshold,
        "threshold_selection": threshold_selection or {"method": "locked"},
        "ranking": {
            "auprc": average_precision(rows),
            "roc_auc": roc_auc(rows),
        },
        "at_threshold": point,
        "operations": {
            "false_alerts_per_1000_user_days": false_alerts_per_1000,
            "alerts_per_day": alerts_per_day,
            "recall_at_top_k_per_day": recall_at_top_k_per_day(rows, top_k),
        },
        "incidents": incident_metrics(rows, threshold),
        "scenario_recall": scenario_recall(rows, threshold),
        "fallback_rates": {
            "feature": fallback_rates(rows, "feature_level"),
            "sequence": fallback_rates(rows, "sequence_level"),
        },
        "coverage": {
            "user_days": len(rows),
            "users": len({row.user_id for row in rows}),
            "days": day_count,
            "positive_user_days": positive_count,
            "negative_user_days": len(rows) - positive_count,
            "no_score_user_days": sum(row.no_score for row in rows),
            "missing_predictions": missing_predictions,
        },
        "confidence_intervals_95": user_block_confidence_intervals(
            rows,
            threshold,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "manifest_checksum": _manifest_checksum(rows, split=split, threshold=threshold),
        "warnings": warnings,
    }
    return result
