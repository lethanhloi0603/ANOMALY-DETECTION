from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from cli.evaluate_metrics import main
from insider_ml.evaluation.metrics import (
    EvaluationInputError,
    PositiveLabel,
    Prediction,
    average_precision,
    evaluate,
    join_predictions_and_labels,
    roc_auc,
    threshold_for_alert_budget,
    threshold_for_validation_quantile,
    threshold_metrics,
)


def _validation_fixture():
    predictions = [
        Prediction(
            "U1",
            date(2010, 6, 1),
            0.9,
            feature_level="PERSON",
            sequence_level="ROLE",
            status="SCORED",
        ),
        Prediction(
            "U2",
            date(2010, 6, 1),
            0.6,
            feature_level="ROLE",
            sequence_level="GLOBAL",
            status="SCORED",
        ),
        Prediction(
            "U1",
            date(2010, 6, 2),
            0.7,
            feature_level="PERSON",
            sequence_level="ROLE",
            status="SCORED",
        ),
        Prediction(
            "U2",
            date(2010, 6, 2),
            0.1,
            feature_level="GLOBAL",
            sequence_level="NO_SCORE",
            status="SCORED",
        ),
    ]
    labels = [
        PositiveLabel("U1", date(2010, 6, 1), incident_id="I1", scenario="S1"),
        PositiveLabel("U1", date(2010, 6, 2), incident_id="I1", scenario="S1"),
    ]
    return join_predictions_and_labels(predictions, labels)


def test_ranking_and_operational_metrics_have_known_values() -> None:
    rows = _validation_fixture()
    assert average_precision(rows) == pytest.approx(1.0)
    assert roc_auc(rows) == pytest.approx(1.0)

    point = threshold_metrics(rows, 0.85)
    assert point == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
        "true_positive": 1,
        "false_positive": 0,
        "true_negative": 2,
        "false_negative": 1,
        "alerts": 1,
    }

    report = evaluate(
        rows,
        split="VALIDATION",
        threshold=0.85,
        top_k=1,
        bootstrap_replicates=20,
        bootstrap_seed=7,
    )
    assert report["ranking"] == {"auprc": 1.0, "roc_auc": 1.0}
    assert report["operations"]["false_alerts_per_1000_user_days"] == 0
    assert report["operations"]["alerts_per_day"] == pytest.approx(0.5)
    assert report["operations"]["recall_at_top_k_per_day"]["value"] == 1.0
    assert report["incidents"]["incident_detection_rate"] == 1.0
    assert report["incidents"]["mean_time_to_detect_days"] == 0
    assert report["scenario_recall"]["S1"]["recall"] == 0.5
    assert report["fallback_rates"]["feature"]["PERSON"]["count"] == 2
    assert report["fallback_rates"]["sequence"]["NO_SCORE"]["count"] == 1
    assert report["confidence_intervals_95"]["auprc"]["block"] == "user"


def test_tied_ranking_equals_prevalence_and_missing_prediction_is_penalized() -> None:
    predictions = [
        Prediction("U1", date(2010, 6, 1), 0.5),
        Prediction("U2", date(2010, 6, 1), 0.5),
        Prediction("U3", date(2010, 6, 1), 0.5),
        Prediction("U4", date(2010, 6, 1), 0.5),
    ]
    labels = [
        PositiveLabel("U1", date(2010, 6, 1)),
        PositiveLabel("U2", date(2010, 6, 1)),
        PositiveLabel("MISSING", date(2010, 6, 1)),
    ]
    rows = join_predictions_and_labels(predictions, labels)
    assert len(rows) == 5
    missing = next(row for row in rows if row.user_id == "MISSING")
    assert missing.missing_prediction is True
    assert missing.no_score is True
    assert missing.risk == 0
    assert roc_auc(rows) == pytest.approx(1 / 3)
    assert average_precision(rows) == pytest.approx((2 / 3) * 0.5 + (1 / 3) * 0.6)


def test_validation_budget_locks_threshold_and_test_cannot_optimize(
    tmp_path: Path,
) -> None:
    rows = _validation_fixture()
    threshold, details = threshold_for_alert_budget(rows, alerts_per_day=0.5)
    assert 0.7 < threshold < 0.9
    assert details["target_alerts"] == 1
    assert threshold_metrics(rows, threshold)["alerts"] == 1

    label_only = PositiveLabel("NO-PREDICTION", date(2010, 6, 30), "I2", "S2")
    rows_with_label_only = join_predictions_and_labels(
        [Prediction(row.user_id, row.day, row.risk) for row in rows if not row.missing_prediction],
        [PositiveLabel(row.user_id, row.day) for row in rows if row.positive] + [label_only],
    )
    threshold_with_label, details_with_label = threshold_for_alert_budget(
        rows_with_label_only,
        alerts_per_day=0.5,
    )
    assert threshold_with_label == threshold
    assert details_with_label["target_alerts"] == details["target_alerts"]

    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "user_id,day,risk\nU1,2010-10-01,0.9\nU2,2010-10-01,0.1\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "user_id,day,incident_id,scenario\nU1,2010-10-01,I1,S1\n",
        encoding="utf-8",
    )
    universe = tmp_path / "universe.csv"
    universe.write_text(
        "user_id,day\nU1,2010-10-01\nU2,2010-10-01\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        main(
            [
                "--predictions-csv",
                str(predictions),
                "--labels-csv",
                str(labels),
                "--positive-only-answer-key",
                "--universe-csv",
                str(universe),
                "--split",
                "TEST",
                "--alert-budget-per-day",
                "1",
            ]
        )


def test_cli_writes_metric_json_from_explicit_positive_only_contract(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "\n".join(
            [
                "user_id,day,risk,feature_level,sequence_level,status",
                "U1,2010-06-01,0.9,PERSON,ROLE,SCORED",
                "U2,2010-06-01,0.1,GLOBAL,NO_SCORE,SCORED",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "user_id,day,incident_id,scenario\nU1,2010-06-01,I1,S1\n",
        encoding="utf-8",
    )
    universe = tmp_path / "universe.csv"
    universe.write_text(
        "user_id,day\nU1,2010-06-01\nU2,2010-06-01\n",
        encoding="utf-8",
    )
    output = tmp_path / "metrics.json"
    exit_code = main(
        [
            "--predictions-csv",
            str(predictions),
            "--labels-csv",
            str(labels),
            "--positive-only-answer-key",
            "--universe-csv",
            str(universe),
            "--split",
            "VALIDATION",
            "--threshold",
            "0.5",
            "--top-k",
            "1",
            "--bootstrap",
            "0",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["ranking"] == {"auprc": 1.0, "roc_auc": 1.0}
    assert result["at_threshold"]["f1"] == 1.0
    assert result["label_contract"]["mode"] == "positive_only_answer_key"
    assert len(result["manifest_checksum"]) == 64


def test_empty_universe_and_bad_risk_are_rejected() -> None:
    with pytest.raises(EvaluationInputError, match="empty"):
        join_predictions_and_labels([], [])
    with pytest.raises(EvaluationInputError, match="risk"):
        Prediction("U1", date(2010, 6, 1), 1.1)


def test_independent_universe_penalizes_missing_negative_predictions() -> None:
    day = date(2010, 6, 1)
    rows = join_predictions_and_labels(
        [Prediction("U1", day, 0.9)],
        [PositiveLabel("U1", day)],
        [("U1", day), ("U2", day)],
    )
    missing_negative = next(row for row in rows if row.user_id == "U2")
    assert missing_negative.missing_prediction is True
    assert missing_negative.positive is False

    threshold, details = threshold_for_validation_quantile(rows, 0.995)
    assert threshold == 0.9
    assert details["quantile"] == 0.995


def test_no_score_never_alerts_or_enters_top_k_at_zero_threshold() -> None:
    rows = join_predictions_and_labels(
        [Prediction("U1", date(2010, 6, 1), None)],
        [PositiveLabel("U1", date(2010, 6, 1), "I1", "S1")],
    )
    report = evaluate(
        rows,
        split="VALIDATION",
        threshold=0,
        top_k=1,
        bootstrap_replicates=0,
        bootstrap_seed=1,
    )
    assert report["at_threshold"]["alerts"] == 0
    assert report["at_threshold"]["false_negative"] == 1
    assert report["operations"]["recall_at_top_k_per_day"]["selected_user_days"] == 0
    assert report["incidents"]["detected_incidents"] == 0


def test_test_rejects_threshold_or_alert_different_from_locked_decision() -> None:
    day = date(2010, 10, 1)
    rows = join_predictions_and_labels(
        [
            Prediction(
                "U1",
                day,
                0.9,
                status="SCORED",
                threshold=0.95,
                is_alert=False,
            )
        ],
        [PositiveLabel("U1", day)],
    )
    with pytest.raises(EvaluationInputError, match="threshold differs"):
        evaluate(
            rows,
            split="TEST",
            threshold=0.9,
            top_k=1,
            bootstrap_replicates=0,
            bootstrap_seed=1,
        )

    inconsistent_rows = join_predictions_and_labels(
        [
            Prediction(
                "U1",
                day,
                0.9,
                status="SCORED",
                threshold=0.8,
                is_alert=False,
            )
        ],
        [PositiveLabel("U1", day)],
    )
    with pytest.raises(EvaluationInputError, match="alert flag is inconsistent"):
        evaluate(
            inconsistent_rows,
            split="TEST",
            threshold=0.8,
            top_k=1,
            bootstrap_replicates=0,
            bootstrap_seed=1,
        )


def test_positive_only_csv_rejects_explicit_negative(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "user_id,day,risk\nU1,2010-06-01,0.9\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "user_id,day,is_positive\nU1,2010-06-01,false\n",
        encoding="utf-8",
    )
    universe = tmp_path / "universe.csv"
    universe.write_text("user_id,day\nU1,2010-06-01\n", encoding="utf-8")
    exit_code = main(
        [
            "--predictions-csv",
            str(predictions),
            "--labels-csv",
            str(labels),
            "--positive-only-answer-key",
            "--universe-csv",
            str(universe),
            "--split",
            "VALIDATION",
            "--threshold",
            "0.5",
        ]
    )
    assert exit_code == 2
