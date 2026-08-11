from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from app.domain.safe_update import (
    FeatureBootstrapSupport,
    ReleaseWindowEntry,
    SafeUpdatePolicy,
    SequenceBootstrapSupport,
    admission_allowed,
    bounded_release_capacity,
    build_personal_calibrator,
    empirical_percentile,
    feature_bootstrap_readiness,
    quarantine_window,
    reference_percentile,
    sequence_bootstrap_readiness,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "framework.v5.json"
LEGACY_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "framework.v4.json"


@pytest.fixture
def framework_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def policy(framework_config: dict[str, object]) -> SafeUpdatePolicy:
    return SafeUpdatePolicy.from_framework(framework_config)


def test_v5_policy_contract(policy: SafeUpdatePolicy) -> None:
    assert policy.policy_version == "framework.v5.safe_update.v1"
    assert policy.admission_max_percentile_exclusive == 0.90
    assert policy.quarantine_days == 30
    assert policy.first_eligible_day_offset == 31
    assert policy.release_interval_days == 7
    assert policy.per_release_influence_cap == 0.02
    assert policy.rolling_influence_cap == 0.10
    assert policy.standalone_min_safe_scores == 200
    assert policy.materialization_enabled is False
    assert policy.activation_requires_zero_pending_legacy_candidates is True


def test_v5_config_does_not_drop_any_v4_top_level_contract() -> None:
    legacy = json.loads(LEGACY_CONFIG_PATH.read_text(encoding="utf-8"))
    current = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert set(legacy).issubset(current)


def test_admission_boundary_is_strict() -> None:
    assert admission_allowed(0.899999, 0.90)
    assert not admission_allowed(0.90, 0.90)
    assert not admission_allowed(0.91, 0.90)


def test_parent_empirical_cdf_boundary() -> None:
    scores = [float(value) for value in range(10)]
    assert empirical_percentile(7.0, scores) == 0.8
    assert empirical_percentile(8.0, scores) == 0.9


def test_quarantine_is_inclusive_through_d_plus_30(
    policy: SafeUpdatePolicy,
) -> None:
    quarantine_until, eligible_on = quarantine_window(date(2026, 1, 1), policy)
    assert quarantine_until == date(2026, 1, 31)
    assert eligible_on == date(2026, 2, 1)


def test_release_capacity_uses_two_percent(policy: SafeUpdatePolicy) -> None:
    assert bounded_release_capacity(
        logical_day=date(2026, 2, 1),
        parent_support=60,
        recent_releases=[],
        policy=policy,
    ) == 1
    assert bounded_release_capacity(
        logical_day=date(2026, 2, 1),
        parent_support=49,
        recent_releases=[],
        policy=policy,
    ) == 0


def test_rolling_cap_uses_fixed_lowest_anchor(policy: SafeUpdatePolicy) -> None:
    recent = [
        ReleaseWindowEntry(date(2026, 1, 5), 60, 1),
        ReleaseWindowEntry(date(2026, 1, 12), 61, 1),
        ReleaseWindowEntry(date(2026, 1, 19), 62, 1),
        ReleaseWindowEntry(date(2026, 1, 26), 63, 1),
    ]
    assert bounded_release_capacity(
        logical_day=date(2026, 2, 1),
        parent_support=64,
        recent_releases=recent,
        policy=policy,
    ) == 1


def test_personal_cdf_is_parent_shrunk_below_200() -> None:
    calibrator = build_personal_calibrator(
        list(reversed([float(value) for value in range(60)])),
        parent_reference_profile_id="parent-id",
        parent_reference_checksum="a" * 64,
        standalone_min_safe_scores=200,
    )
    assert calibrator["method"] == "parent_shrunk_ecdf.v1"
    result = reference_percentile(
        30.0,
        calibrator,
        parent_sorted_scores=[float(value) for value in range(100)],
    )
    assert result == pytest.approx(0.30 * (31 / 60) + 0.70 * (31 / 100))


def test_personal_cdf_becomes_independent_at_200() -> None:
    calibrator = build_personal_calibrator(
        [float(value) for value in range(200)],
        parent_reference_profile_id="parent-id",
        parent_reference_checksum="b" * 64,
        standalone_min_safe_scores=200,
    )
    assert calibrator["method"] == "empirical_cdf.v1"
    assert len(calibrator["sorted_scores"]) == 200


def test_feature_bootstrap_boundaries(framework_config: dict[str, object]) -> None:
    ready = feature_bootstrap_readiness(
        FeatureBootstrapSupport(60, 90, 60, 0.90, 40),
        framework_config,
    )
    blocked = feature_bootstrap_readiness(
        FeatureBootstrapSupport(59, 89, 59, 0.899, 39),
        framework_config,
    )
    assert ready.ready
    assert blocked.reason_codes == (
        "P_ACTIVE_DAYS_LOW",
        "P_SPAN_DAYS_LOW",
        "P_ROLE_TENURE_LOW",
        "P_COVERAGE_LOW",
        "P_FEATURE_SUPPORT_LOW",
    )


def test_sequence_bootstrap_boundaries(framework_config: dict[str, object]) -> None:
    ready = sequence_bootstrap_readiness(
        SequenceBootstrapSupport(60, 90, 60, 1500),
        framework_config,
    )
    blocked = sequence_bootstrap_readiness(
        SequenceBootstrapSupport(59, 89, 59, 1499),
        framework_config,
    )
    assert ready.ready
    assert blocked.reason_codes == (
        "SP_DAYS_LOW",
        "SP_TRANSITIONS_LOW",
        "SP_SPAN_DAYS_LOW",
        "SP_ROLE_TENURE_LOW",
    )
