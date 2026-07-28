from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from insider_ml.preprocessing import (
    OrderedEvent,
    PCUsage,
    classify_pc_context,
    cross_source_transition_features,
    email_metadata_features,
    file_metadata_features,
    fit_pc_profiles,
    fit_robust_feature_scaler,
    http_metadata_features,
    new_domain_count_30d,
    normalize_hostname,
)


def test_pc_context_uses_distinct_train_user_days_and_locked_tie_break() -> None:
    usage = [
        PCUsage(user_id=f"U{user}", pc="SHARED-1", day=date(2010, 1, user))
        for user in range(1, 6)
    ]
    usage.extend(
        [
            PCUsage(user_id="OWNER", pc="OWNED-1", day=date(2010, 1, day))
            for day in range(1, 6)
        ]
    )
    usage.append(PCUsage(user_id="VISITOR", pc="OWNED-1", day=date(2010, 1, 1)))

    profiles = fit_pc_profiles(usage)

    assert profiles["SHARED-1"].is_shared is True
    assert classify_pc_context(profiles, user_id="U1", pc="SHARED-1") == "SHARED"
    assert classify_pc_context(profiles, user_id="OWNER", pc="OWNED-1") == "OWN"
    assert classify_pc_context(profiles, user_id="VISITOR", pc="OWNED-1") == "FOREIGN"
    assert classify_pc_context(profiles, user_id="U1", pc="UNSEEN") == "UNKNOWN"


def test_domain_novelty_is_past_only_and_masks_insufficient_history() -> None:
    prior = ["HTTP://EXAMPLE.TEST./a"] * 19
    assert new_domain_count_30d(["https://new.test"], prior) is None

    prior.append("https://known.test/path")
    assert new_domain_count_30d(
        ["https://known.test/other", "https://new.test/a", "https://new.test/b"],
        prior,
    ) == 1
    assert normalize_hostname("HTTP://EXAMPLE.TEST./a") == "example.test"


def test_metadata_replacements_do_not_require_content() -> None:
    start = datetime(2010, 1, 1, 9, 0, tzinfo=UTC)
    file_values = file_metadata_features(
        ["a.pdf", "long-name.doc", "third.xls"],
        [start, start + timedelta(minutes=3), start + timedelta(minutes=20)],
    )
    assert file_values["file_copy_burst_count_5m"] == 1
    assert file_values["file_copy_interarrival_min_minutes"] == 3
    assert file_values["filename_length_max"] == 13
    assert file_values["filename_length_std"] == pytest.approx(
        np.std([5, 13, 9], ddof=0)
    )
    assert file_values["extension_length_mean"] == pytest.approx(3)
    assert file_values["extensionless_file_count"] == 0
    assert file_values["distinct_filename_stem_count"] == 3
    assert file_values["repeated_extension_copy_count"] == 0
    assert file_values["extension_present_ratio"] == 1

    http_values = http_metadata_features(
        [
            "https://example.test/a/b?q=1",
            "https://example.test/a/c?q=2&lang=vi",
        ]
    )
    assert http_values["distinct_hostname_count"] == 1
    assert http_values["https_request_count"] == 2
    assert http_values["distinct_query_key_count"] == 2
    assert http_values["repeated_hostname_request_count"] == 1
    assert http_values["url_path_depth_mean"] == 2
    assert http_values["distinct_url_path_count"] == 2

    email_values = email_metadata_features([1, 3], [100.0, 300.0])
    assert email_values == {"recipient_count_mean": 2.0, "email_size_std": 100.0}


def test_generic_transition_features_replace_scenario_chains() -> None:
    start = datetime(2010, 1, 1, 9, 0, tzinfo=UTC)
    values = cross_source_transition_features(
        [
            OrderedEvent(start, "LOGON", "1"),
            OrderedEvent(start + timedelta(minutes=1), "DEVICE", "2"),
            OrderedEvent(start + timedelta(minutes=2), "FILE", "3"),
            OrderedEvent(start + timedelta(minutes=12), "FILE", "4"),
        ]
    )
    assert values["source_transition_count"] == 2
    assert values["distinct_source_transition_count"] == 2
    assert values["source_transition_entropy"] == pytest.approx(1.0)
    assert values["inter_event_gap_p95_minutes"] == 10


def test_train_only_robust_scaler_masks_missing_values() -> None:
    values = np.zeros((2, 30, 128), dtype=np.float64)
    mask = np.ones_like(values, dtype=bool)
    values[1, :, 0] = 10
    mask[0, 0, 1] = False

    scaler = fit_robust_feature_scaler(values, mask)
    transformed = scaler.transform(values, mask)

    assert scaler.checksum
    assert transformed.shape == values.shape
    assert transformed[0, 0, 1] == 0
    assert np.isfinite(transformed).all()


def test_extension_features_do_not_use_an_unversioned_category_mapping() -> None:
    path = Path(__file__).resolve().parents[1] / "config" / "feature128.v5.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    policy = catalog["extension_category_policy"]
    names = {feature["name"] for feature in catalog["features"]}

    assert policy == {
        "category_mapping_enabled": False,
        "mapping_version": None,
        "fallback": "neutral_extension_metadata_only",
    }
    assert not any("extension_category" in name for name in names)
    assert {
        "distinct_extension_count",
        "file_extension_entropy",
        "extension_length_mean",
        "extension_length_max",
    } <= names
