"""Pure policy primitives for poisoning-resistant Personal reference updates.

This module deliberately contains no database or HTTP concerns.  Admission,
quarantine, calibration, and release-cap boundaries therefore have one
deterministic implementation that can be reused by services and tests.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import median
from typing import Any


@dataclass(frozen=True, slots=True)
class SafeUpdatePolicy:
    policy_version: str
    admission_max_percentile_exclusive: float
    quarantine_days: int
    first_eligible_day_offset: int
    release_interval_days: int
    per_release_influence_cap: float
    rolling_window_days: int
    rolling_influence_cap: float
    standalone_min_safe_scores: int

    @classmethod
    def from_framework(cls, config: Mapping[str, Any]) -> SafeUpdatePolicy:
        raw = config["safe_personalized_update"]
        admission = raw["admission"]
        release = raw["release"]
        calibrator = raw["personal_calibrator"]
        policy = cls(
            policy_version=str(raw["policy_version"]),
            admission_max_percentile_exclusive=float(
                admission["max_parent_empirical_cdf_exclusive"]
            ),
            quarantine_days=int(raw["quarantine_days"]),
            first_eligible_day_offset=int(raw["first_eligible_day_offset"]),
            release_interval_days=int(release["interval_days"]),
            per_release_influence_cap=float(release["per_release_influence_cap"]),
            rolling_window_days=int(release["rolling_window_days"]),
            rolling_influence_cap=float(release["rolling_influence_cap"]),
            standalone_min_safe_scores=int(
                calibrator["standalone_min_safe_scores"]
            ),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if not self.policy_version:
            raise ValueError("policy_version cannot be empty")
        if not 0 < self.admission_max_percentile_exclusive < 1:
            raise ValueError("admission percentile must be in (0, 1)")
        if self.quarantine_days < 1:
            raise ValueError("quarantine_days must be positive")
        if self.first_eligible_day_offset != self.quarantine_days + 1:
            raise ValueError(
                "first_eligible_day_offset must equal quarantine_days + 1"
            )
        if self.release_interval_days < 1:
            raise ValueError("release_interval_days must be positive")
        if not 0 < self.per_release_influence_cap <= 1:
            raise ValueError("per-release influence cap must be in (0, 1]")
        if self.rolling_window_days < self.release_interval_days:
            raise ValueError("rolling window cannot be shorter than release interval")
        if not 0 < self.rolling_influence_cap <= 1:
            raise ValueError("rolling influence cap must be in (0, 1]")
        if self.per_release_influence_cap > self.rolling_influence_cap:
            raise ValueError("per-release influence cap cannot exceed rolling cap")
        if self.standalone_min_safe_scores < 1:
            raise ValueError("standalone Personal support must be positive")


@dataclass(frozen=True, slots=True)
class ReleaseWindowEntry:
    release_day: date
    parent_support: int
    applied_candidate_count: int


@dataclass(frozen=True, slots=True)
class BootstrapDecision:
    ready: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FeatureBootstrapSupport:
    active_days: int
    span_days: int
    active_days_current_role: int
    coverage: float
    min_feature_observations: int


@dataclass(frozen=True, slots=True)
class SequenceBootstrapSupport:
    sequence_days: int
    span_days: int
    sequence_days_current_role: int
    transitions: int


def _finite_scores(
    values: Sequence[float],
    *,
    field_name: str,
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{field_name} must contain only finite values")
    return normalized


def _finite_sorted_scores(
    values: Sequence[float],
    *,
    field_name: str,
) -> tuple[float, ...]:
    normalized = _finite_scores(values, field_name=field_name)
    if normalized != tuple(sorted(normalized)):
        raise ValueError(f"{field_name} must be sorted")
    return normalized


def empirical_percentile(
    raw_score: float,
    sorted_scores: Sequence[float],
) -> float:
    """Return the right-continuous empirical CDF used by reference profiles."""

    score = float(raw_score)
    if not math.isfinite(score):
        raise ValueError("raw_score must be finite")
    reference = _finite_sorted_scores(sorted_scores, field_name="sorted_scores")
    return bisect_right(reference, score) / len(reference)


def admission_allowed(
    percentile: float,
    threshold_exclusive: float,
) -> bool:
    """Apply the strict admission boundary; equality is rejected."""

    value = float(percentile)
    threshold = float(threshold_exclusive)
    return (
        math.isfinite(value)
        and math.isfinite(threshold)
        and 0 <= value < threshold < 1
    )


def reference_percentile(
    raw_score: float,
    calibrator: Mapping[str, Any],
    *,
    parent_sorted_scores: Sequence[float] | None = None,
) -> float:
    """Evaluate either a legacy empirical or v5 parent-shrunk CDF."""

    method = calibrator.get("method")
    if method == "empirical_cdf.v1":
        return empirical_percentile(raw_score, calibrator["sorted_scores"])
    if method != "parent_shrunk_ecdf.v1":
        raise ValueError(f"unsupported calibrator method: {method!r}")
    if parent_sorted_scores is None:
        raise ValueError("parent-shrunk CDF requires immutable parent scores")

    personal = _finite_sorted_scores(
        calibrator["personal_sorted_scores"],
        field_name="personal_sorted_scores",
    )
    target = int(calibrator["standalone_target_observations"])
    if target < 1 or len(personal) >= target:
        raise ValueError("invalid support for parent-shrunk CDF")

    personal_q = empirical_percentile(raw_score, personal)
    parent_q = empirical_percentile(raw_score, parent_sorted_scores)
    personal_weight = len(personal) / target
    return personal_weight * personal_q + (1.0 - personal_weight) * parent_q


def build_personal_calibrator(
    personal_scores: Sequence[float],
    *,
    parent_reference_profile_id: str,
    parent_reference_checksum: str,
    standalone_min_safe_scores: int,
) -> dict[str, Any]:
    """Build an immutable Personal CDF with a pinned parent below support 200."""

    scores = tuple(
        sorted(_finite_scores(personal_scores, field_name="personal_scores"))
    )
    if standalone_min_safe_scores < 1:
        raise ValueError("standalone_min_safe_scores must be positive")
    if len(scores) >= standalone_min_safe_scores:
        return {
            "method": "empirical_cdf.v1",
            "sorted_scores": list(scores),
        }
    if not parent_reference_profile_id or not parent_reference_checksum:
        raise ValueError("parent identity and checksum are required")
    return {
        "method": "parent_shrunk_ecdf.v1",
        "personal_sorted_scores": list(scores),
        "parent_reference_profile_id": parent_reference_profile_id,
        "parent_reference_checksum": parent_reference_checksum,
        "standalone_target_observations": standalone_min_safe_scores,
    }


def robust_score_statistics(scores: Sequence[float]) -> dict[str, float | int]:
    """Return the reference location/scale contract without mutating a parent."""

    values = tuple(sorted(_finite_scores(scores, field_name="scores")))
    location = median(values)
    deviations = tuple(sorted(abs(value - location) for value in values))
    mad = median(deviations)
    return {
        "location": float(location),
        "scale": max(1.4826 * float(mad), 1e-6),
        "observation_count": len(values),
    }


def quarantine_window(
    candidate_day: date,
    policy: SafeUpdatePolicy,
) -> tuple[date, date]:
    """Return inclusive quarantine end D+30 and first eligible day D+31."""

    return (
        candidate_day + timedelta(days=policy.quarantine_days),
        candidate_day + timedelta(days=policy.first_eligible_day_offset),
    )


def bounded_release_capacity(
    *,
    logical_day: date,
    parent_support: int,
    recent_releases: Sequence[ReleaseWindowEntry],
    policy: SafeUpdatePolicy,
) -> int:
    """Return the release count allowed by both 2% and rolling 10% caps."""

    if parent_support < 1:
        return 0
    earliest = logical_day - timedelta(days=policy.rolling_window_days - 1)
    recent = tuple(
        release
        for release in recent_releases
        if earliest <= release.release_day <= logical_day
    )
    per_release_limit = math.floor(
        policy.per_release_influence_cap * parent_support
    )
    if per_release_limit < 1:
        return 0

    anchor_support = min(
        [parent_support, *(release.parent_support for release in recent)]
    )
    rolling_limit = math.floor(policy.rolling_influence_cap * anchor_support)
    rolling_used = sum(release.applied_candidate_count for release in recent)
    return min(per_release_limit, max(rolling_limit - rolling_used, 0))


def feature_bootstrap_readiness(
    support: FeatureBootstrapSupport,
    config: Mapping[str, Any],
) -> BootstrapDecision:
    cfg = config["readiness"]["feature"]["person"]
    reasons: list[str] = []
    if support.active_days < int(cfg["min_active_days"]):
        reasons.append("P_ACTIVE_DAYS_LOW")
    if support.span_days < int(cfg["min_span_days"]):
        reasons.append("P_SPAN_DAYS_LOW")
    if support.active_days_current_role < int(
        cfg["min_active_days_in_current_role"]
    ):
        reasons.append("P_ROLE_TENURE_LOW")
    if support.coverage < float(cfg["min_feature_coverage_ratio"]):
        reasons.append("P_COVERAGE_LOW")
    if support.min_feature_observations < int(
        cfg["min_observations_per_used_feature"]
    ):
        reasons.append("P_FEATURE_SUPPORT_LOW")
    return BootstrapDecision(not reasons, tuple(reasons))


def sequence_bootstrap_readiness(
    support: SequenceBootstrapSupport,
    config: Mapping[str, Any],
) -> BootstrapDecision:
    cfg = config["readiness"]["sequence"]["person"]
    reasons: list[str] = []
    if support.sequence_days < int(cfg["min_sequence_days"]):
        reasons.append("SP_DAYS_LOW")
    if support.transitions < int(cfg["min_transitions"]):
        reasons.append("SP_TRANSITIONS_LOW")
    if support.span_days < int(cfg["min_span_days"]):
        reasons.append("SP_SPAN_DAYS_LOW")
    if support.sequence_days_current_role < int(
        cfg["min_sequence_days_in_current_role"]
    ):
        reasons.append("SP_ROLE_TENURE_LOW")
    return BootstrapDecision(not reasons, tuple(reasons))
