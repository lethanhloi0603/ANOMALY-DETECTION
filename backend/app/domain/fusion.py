"""Calibration-score fusion for independently gated branches."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .readiness import _resolve_config


@dataclass(frozen=True)
class FusionResult:
    risk: float | None
    feature_weight: float
    sequence_weight: float
    available_branches: tuple[str, ...]
    reason_codes: tuple[str, ...]
    config_version: str

    @property
    def can_alert(self) -> bool:
        return self.risk is not None

    @property
    def is_no_score(self) -> bool:
        return self.risk is None


def _validated_score(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return score


def _validated_weight(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    weight = float(value)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return weight


def fuse_scores(
    *,
    q_feature: float | None,
    q_sequence: float | None,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> FusionResult:
    """Fuse calibrated scores and renormalize weights over present branches.

    ``None`` means that a branch returned ``NO_SCORE``.  When only one branch
    is present its normalized weight is 1.0, so its calibrated percentile is
    preserved exactly.  No alert can be produced when both branches are
    absent.
    """

    resolved = _resolve_config(config, config_path)
    q_f = _validated_score(q_feature, "q_feature")
    q_s = _validated_score(q_sequence, "q_sequence")
    feature_weight = _validated_weight(resolved["fusion"]["feature_weight"], "feature_weight")
    sequence_weight = _validated_weight(resolved["fusion"]["sequence_weight"], "sequence_weight")

    if q_f is None and q_s is None:
        return FusionResult(
            risk=None,
            feature_weight=0.0,
            sequence_weight=0.0,
            available_branches=(),
            reason_codes=("FUSION_NO_BRANCH_SCORE",),
            config_version=str(resolved["version"]),
        )

    present_feature_weight = feature_weight if q_f is not None else 0.0
    present_sequence_weight = sequence_weight if q_s is not None else 0.0
    total_weight = present_feature_weight + present_sequence_weight
    if total_weight <= 0.0:
        raise ValueError("At least one available branch must have positive weight")

    normalized_feature_weight = present_feature_weight / total_weight
    normalized_sequence_weight = present_sequence_weight / total_weight
    risk = (q_f or 0.0) * normalized_feature_weight + (q_s or 0.0) * normalized_sequence_weight

    branches: list[str] = []
    reasons: list[str] = []
    if q_f is not None:
        branches.append("FEATURE")
    else:
        reasons.append("FEATURE_NO_SCORE")
    if q_s is not None:
        branches.append("SEQUENCE")
    else:
        reasons.append("SEQUENCE_NO_SCORE")

    return FusionResult(
        risk=risk,
        feature_weight=normalized_feature_weight,
        sequence_weight=normalized_sequence_weight,
        available_branches=tuple(branches),
        reason_codes=tuple(reasons),
        config_version=str(resolved["version"]),
    )
