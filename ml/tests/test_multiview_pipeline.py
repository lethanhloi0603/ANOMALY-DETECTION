from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from model import MultiViewTCNTransformerAutoencoder
from evaluate_multiview import evaluate_one
from train_multiview import LazyCalendarWindowDataset, calibrate_personal, train_global
from multiview_prepare_fast import choose_holdout_users
from multiview_features import (
    SOURCE_TO_ID,
    TOKEN_TO_ID,
    attach_role_context,
    calendarize_multiview,
    configure_feature_rules,
    count_external,
    is_after_hour,
    make_multiview_day_inputs,
    make_multiview_model_inputs,
)


class MultiviewPipelineTests(unittest.TestCase):
    def setUp(self):
        configure_feature_rules()

    def test_calendar_days_are_zero_filled_and_windowed_daily(self):
        frame = pd.DataFrame([
            {
                "user": "U1",
                "date": "2010-01-01",
                "role": "Engineer",
                "department": "R&D",
                "total_events": 3,
                "event_sequence": json.dumps([TOKEN_TO_ID["DAY_START"], TOKEN_TO_ID["LOGON"], TOKEN_TO_ID["DAY_END"]]),
                "source_sequence": json.dumps([SOURCE_TO_ID["unknown"], SOURCE_TO_ID["logon"], SOURCE_TO_ID["unknown"]]),
                "time_gap_sequence": json.dumps([0, 0, 0]),
            },
            {
                "user": "U1",
                "date": "2010-01-03",
                "role": "Engineer",
                "department": "R&D",
                "total_events": 5,
                "event_sequence": json.dumps([TOKEN_TO_ID["DAY_START"], TOKEN_TO_ID["FILE_ACCESS"], TOKEN_TO_ID["DAY_END"]]),
                "source_sequence": json.dumps([SOURCE_TO_ID["unknown"], SOURCE_TO_ID["file"], SOURCE_TO_ID["unknown"]]),
                "time_gap_sequence": json.dumps([0, 0, 0]),
            },
        ])
        calendar = calendarize_multiview(frame)
        self.assertEqual(calendar["date"].tolist(), ["2010-01-01", "2010-01-02", "2010-01-03"])
        self.assertEqual(float(calendar.loc[1, "total_events"]), 0.0)

        counts, tokens, sources, gaps, metas = make_multiview_model_inputs(
            calendar,
            ["total_events"],
            window_size=3,
            max_events_per_day=8,
            allow_padding=True,
        )
        self.assertEqual(counts.shape, (3, 3, 1))
        self.assertEqual(len(metas), 3)
        self.assertEqual(metas[1]["date"], "2010-01-02")
        self.assertEqual(float(counts[1, -1, 0]), 0.0)
        self.assertEqual(tokens.shape, sources.shape)
        self.assertEqual(tokens.shape, gaps.shape)

    def test_lazy_windows_match_eager_windows_without_materializing_all_windows(self):
        frame = pd.DataFrame([
            {
                "user": "U1", "date": f"2010-01-0{day}", "role": "Engineer", "department": "R&D",
                "total_events": day,
                "event_sequence": json.dumps([TOKEN_TO_ID["DAY_START"], TOKEN_TO_ID["LOGON"], TOKEN_TO_ID["DAY_END"]]),
                "source_sequence": json.dumps([SOURCE_TO_ID["unknown"], SOURCE_TO_ID["logon"], SOURCE_TO_ID["unknown"]]),
                "time_gap_sequence": json.dumps([0, day, 0]),
                "split": "train", "label_day": 0, "scenario": "",
            }
            for day in range(1, 5)
        ])
        eager = make_multiview_model_inputs(frame, ["total_events"], 3, max_events_per_day=8)
        day_inputs = make_multiview_day_inputs(frame, ["total_events"], 3, max_events_per_day=8)
        count, tokens, sources, gaps, slices, metas = day_inputs
        lazy = LazyCalendarWindowDataset(count, tokens, sources, gaps, slices, 3)
        self.assertEqual(len(lazy), len(eager[-1]))
        self.assertLess(lazy.storage_bytes, sum(array.nbytes for array in eager[:4]))
        for index in range(len(lazy)):
            lazy_window = lazy[index]
            for lazy_tensor, eager_array in zip(lazy_window, eager[:4]):
                np.testing.assert_allclose(lazy_tensor.numpy(), eager_array[index])
        self.assertEqual(metas[-1]["date"], eager[-1][-1]["date"])

    def test_scientific_holdout_is_label_blind(self):
        frame = pd.DataFrame([
            {"user": user, "date": f"2010-01-{day:02d}", "role": role, "label_day": label}
            for user, role, label in [("U1", "A", 1), ("U2", "A", 0), ("U3", "B", 1), ("U4", "B", 0)]
            for day in range(1, 4)
        ])
        selected = choose_holdout_users(frame, 2, 1, "role-stratified", 42)
        flipped = frame.copy()
        flipped["label_day"] = 1 - flipped["label_day"]
        selected_after_flip = choose_holdout_users(flipped, 2, 1, "role-stratified", 42)
        self.assertEqual(selected, selected_after_flip)

    def test_ldap_fallback_never_uses_future_snapshot(self):
        events = pd.DataFrame([
            {"user": "U1", "date": "2010-01-15", "event_uid": "x"},
            {"user": "U2", "date": "2010-01-15", "event_uid": "y"},
        ])
        context = pd.DataFrame([
            {
                "user": "U1", "valid_month": "2009-12", "role": "Analyst", "business_unit": "B",
                "functional_unit": "F", "department": "D", "team": "T", "supervisor": "S",
            },
            {
                "user": "U1", "valid_month": "2010-02", "role": "Manager", "business_unit": "B",
                "functional_unit": "F", "department": "D2", "team": "T2", "supervisor": "S2",
            },
            {
                "user": "U2", "valid_month": "2010-02", "role": "FutureOnly", "business_unit": "B",
                "functional_unit": "F", "department": "D", "team": "T", "supervisor": "S",
            },
        ])
        attached = attach_role_context(events, context)
        self.assertEqual(attached.loc[attached["user"] == "U1", "role"].iloc[0], "Analyst")
        self.assertEqual(attached.loc[attached["user"] == "U2", "role"].iloc[0], "UNKNOWN")

    def test_email_domain_and_after_hours_rules_are_auditable(self):
        self.assertEqual(count_external("a@dtaa.com;b@example.com;missing-domain"), 1)
        self.assertEqual(is_after_hour(pd.Timestamp("2010-01-01 02:00:00")), 0)

    def test_raw_sequence_model_forward_shape(self):
        model = MultiViewTCNTransformerAutoencoder(
            count_dim=4,
            sequence_dim=4,
            window_size=3,
            vocab_size=len(TOKEN_TO_ID),
            source_vocab_size=len(SOURCE_TO_ID),
            max_events_per_day=8,
            hidden_dim=16,
            kernel_size=3,
            dropout=0.0,
            transformer_layers=1,
            heads=4,
        )
        count = torch.zeros((2, 3, 4), dtype=torch.float32)
        tokens = torch.zeros((2, 3, 8), dtype=torch.long)
        tokens[:, :, :2] = torch.tensor([TOKEN_TO_ID["DAY_START"], TOKEN_TO_ID["DAY_END"]])
        sources = torch.full((2, 3, 8), SOURCE_TO_ID["unknown"], dtype=torch.long)
        gaps = torch.zeros((2, 3, 8), dtype=torch.float32)
        reconstruction, target = model(count, tokens, sources, gaps)
        self.assertEqual(tuple(reconstruction.shape), (2, 3, 8))
        self.assertEqual(tuple(target.shape), (2, 3, 8))
        self.assertTrue(np.isfinite(reconstruction.detach().numpy()).all())

    def test_evaluation_uses_validation_threshold_and_test_only_metrics(self):
        rows = []
        for index in range(10):
            rows.append({
                "split": "validation",
                "score": index / 10,
                "label_day": 0,
                "user": "V",
                "date": f"2010-01-{index + 1:02d}",
                "scenario": "",
            })
        for index in range(10):
            rows.append({
                "split": "test",
                "score": 2.0 if index == 9 else index / 20,
                "label_day": int(index == 9),
                "user": "T",
                "date": f"2010-02-{index + 1:02d}",
                "scenario": "1" if index == 9 else "",
            })
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scores.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            metrics, scenarios = evaluate_one("A2", path, [0.95])
            self.assertEqual(len(metrics), 1)
            self.assertEqual(metrics[0]["positive_user_days"], 1)
            self.assertEqual(metrics[0]["recall"], 1.0)
            self.assertEqual(metrics[0]["incident_recall"], 1.0)
            self.assertEqual(metrics[0]["false_positive_rate"], 0.0)
            self.assertEqual(scenarios[0]["scenario"], "1")

    def test_global_training_smoke_uses_temporal_validation(self):
        rows = []
        for user_index, user in enumerate(["U1", "U2"]):
            for day_index, date in enumerate(pd.date_range("2010-01-01", periods=20, freq="D")):
                rows.append({
                    "user": user,
                    "date": date.strftime("%Y-%m-%d"),
                    "role": "Engineer",
                    "department": "R&D",
                    "total_events": day_index + 1 + user_index,
                    "event_sequence": json.dumps([TOKEN_TO_ID["DAY_START"], TOKEN_TO_ID["LOGON"], TOKEN_TO_ID["DAY_END"]]),
                    "source_sequence": json.dumps([SOURCE_TO_ID["unknown"], SOURCE_TO_ID["logon"], SOURCE_TO_ID["unknown"]]),
                    "time_gap_sequence": json.dumps([0, 0, 0]),
                    "label_day": int(day_index == 19),
                    "scenario": "1" if day_index == 19 else "",
                })
        frame = calendarize_multiview(pd.DataFrame(rows))
        args = SimpleNamespace(
            scope="global",
            train_fraction=0.60,
            validation_fraction=0.20,
            window_size=5,
            max_events_per_day=8,
            hidden_dim=16,
            kernel_size=3,
            dropout=0.0,
            transformer_layers=1,
            heads=4,
            view_mode="count-sequence",
            batch_size=8,
            epochs=1,
            lr=1e-3,
            anomaly_quantile=0.95,
            min_role_users=1,
            min_role_user_days=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = train_global(frame, active_days=20, model_dir=Path(temp_dir), args=args, started=time.time())
            self.assertTrue(result["ok"])
            metadata = json.loads((Path(temp_dir) / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["threshold_source"], "validation")
            self.assertFalse(metadata["labels_used_for_fit"])
            scores = pd.read_csv(Path(temp_dir) / "all_scores.csv")
            self.assertEqual(set(scores["split"]), {"train", "validation", "test"})

            personal_dir = Path(temp_dir) / "personalized" / "U1"
            personal_dir.mkdir(parents=True)
            personal_args = SimpleNamespace(
                scope="personalized",
                user_id="U1",
                global_model_dir=temp_dir,
                batch_size=8,
                min_safe_days=5,
                safe_history_days=20,
                update_delay_days=2,
                update_every_safe_days=3,
                update_every_calendar_days=3,
                force_update=True,
                max_events_per_day=8,
                window_size=5,
                view_mode="count-sequence",
            )
            personal_result = calibrate_personal(
                frame[frame["user"] == "U1"].copy(),
                active_days=20,
                model_dir=personal_dir,
                args=personal_args,
            )
            self.assertTrue(personal_result["eligible"])
            self.assertTrue(personal_result["updated"])
            personal_metadata = json.loads((personal_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(personal_metadata["calibration_type"], "threshold_only_no_personal_neural_model")
            self.assertFalse((personal_dir / "model.pt").exists())


if __name__ == "__main__":
    unittest.main()
