from __future__ import annotations

from datetime import date

import pytest

from insider_ml.temporal import (
    TemporalBaseline,
    build_temporal_feature_values,
    circular_distance_minutes,
    cyclic_time_components,
    fit_temporal_baseline,
    select_temporal_baseline,
    summarize_temporal_deviation,
    unusual_time_relative_to_baseline,
)


def test_cyclic_time_wraps_at_midnight() -> None:
    midnight = cyclic_time_components(0)
    next_midnight = cyclic_time_components(1440)
    noon = cyclic_time_components(720)

    assert midnight == pytest.approx(next_midnight)
    assert midnight == pytest.approx((0.0, 1.0), abs=1e-7)
    assert noon == pytest.approx((0.0, -1.0), abs=1e-7)
    assert circular_distance_minutes(1435, 5) == pytest.approx(10.0)


def test_temporal_baseline_scores_user_relative_deviation() -> None:
    baseline = fit_temporal_baseline(
        [475, 480, 485, 478, 482],
        level="PERSON",
        scope_key="person:U001",
        fitted_through=date(2010, 5, 31),
    )

    normal = unusual_time_relative_to_baseline(481, baseline)
    unusual = unusual_time_relative_to_baseline(60, baseline)
    mean_score, max_score = summarize_temporal_deviation([481, 60], baseline)

    assert normal == pytest.approx(0.0)
    assert unusual > 100
    assert max_score == pytest.approx(unusual)
    assert 0 < mean_score < max_score


def test_temporal_baseline_falls_back_person_role_global_using_past_only() -> None:
    future_person = TemporalBaseline(
        level="PERSON",
        scope_key="person:U001",
        fitted_through=date(2010, 6, 1),
        center_minute=480,
        distance_location=5,
        distance_scale=2,
        observation_count=30,
    )
    role = TemporalBaseline(
        level="ROLE",
        scope_key="role:ENGINEER",
        fitted_through=date(2010, 5, 31),
        center_minute=500,
        distance_location=10,
        distance_scale=5,
        observation_count=300,
    )

    selected = select_temporal_baseline(
        date(2010, 6, 1),
        person=future_person,
        role=role,
    )

    assert selected is role


def test_temporal_baseline_requires_a_past_reference() -> None:
    with pytest.raises(ValueError, match="past-only"):
        select_temporal_baseline(date(2010, 6, 1))


def test_temporal_feature_builder_returns_feature128_v5_names_and_missing_values() -> None:
    baseline = fit_temporal_baseline(
        [475, 480, 485, 478, 482],
        level="PERSON",
        scope_key="person:U001",
        fitted_through=date(2010, 5, 31),
    )
    groups = {
        "FIRST_LOGON": [481],
        "LAST_LOGOFF": [1060],
        "LOGON_SESSION": [481, 1060],
        "DEVICE": [],
        "FILE": [60],
        "HTTP": [482],
        "EMAIL": [],
        "ALL": [481, 60, 482],
    }
    references = {
        name: baseline
        for name in groups
        if groups[name]
    }

    values = build_temporal_feature_values(groups, references)

    assert set(values) == {
        "first_logon_time_deviation",
        "last_logoff_time_deviation",
        "session_time_deviation_max",
        "device_time_deviation_mean",
        "file_time_deviation_mean",
        "http_time_deviation_mean",
        "email_time_deviation_mean",
        "unusual_time_relative_to_baseline_mean",
        "unusual_time_relative_to_baseline_max",
    }
    assert values["device_time_deviation_mean"] is None
    assert values["email_time_deviation_mean"] is None
    assert values["file_time_deviation_mean"] > values["http_time_deviation_mean"]
