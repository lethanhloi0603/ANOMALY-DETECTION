"""Deterministic Person -> Role -> Global readiness selection.

The framework scores one ``(user_id, score_date)`` at a time.  Feature and
sequence readiness are deliberately evaluated independently.  Every support
snapshot must be strictly historical: ``support_as_of < score_date``.

Configuration is loaded lazily from the configured versioned framework file so importing
this module remains safe while a deployment is being bootstrapped.  If the
file does not exist, the versioned framework defaults below are used.  A
present but malformed file is an error; silently ignoring bad configuration
would make reference selection non-reproducible.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.settings import settings

DEFAULT_CONFIG_PATH = settings.framework_config_path

DEFAULT_FRAMEWORK_CONFIG: dict[str, Any] = {
    "version": "framework.v5",
    "readiness": {
        "feature": {
            "person": {
                "min_active_days": 60,
                "min_span_days": 90,
                "min_role_active_days": 60,
                "min_coverage": 0.90,
                "min_feature_observations": 40,
                "max_last_active_gap_days": 30,
            },
            "role": {
                "min_peer_users": 15,
                "min_peer_user_days": 300,
                "min_recent_user_days": 100,
                "min_coverage": 0.90,
                "min_feature_observations": 200,
            },
            "global": {
                "min_users": 200,
                "min_user_days": 10_000,
                "min_coverage": 0.90,
                "require_train_fitted": True,
            },
        },
        "sequence": {
            "current_day": {"min_seq_len": 2},
            "person": {
                "min_sequence_days": 60,
                "min_transitions": 1_500,
                "min_span_days": 90,
                "min_role_sequence_days": 60,
                "max_last_active_gap_days": 30,
            },
            "role": {
                "min_peer_users": 15,
                "min_sequence_days": 300,
                "min_transitions": 10_000,
                "min_recent_transitions": 2_000,
            },
            "global": {
                "min_users": 200,
                "min_sequence_days": 10_000,
                "min_transitions": 100_000,
                "require_train_fitted": True,
            },
        },
    },
    "fusion": {
        "feature_weight": 0.5,
        "sequence_weight": 0.5,
    },
}


class Branch(StrEnum):
    FEATURE = "FEATURE"
    SEQUENCE = "SEQUENCE"


class ReferenceLevel(StrEnum):
    PERSON = "PERSON"
    ROLE = "ROLE"
    GLOBAL = "GLOBAL"
    NO_SCORE = "NO_SCORE"


@dataclass(frozen=True)
class ScopeEvaluation:
    level: ReferenceLevel
    eligible: bool
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReadinessDecision:
    branch: Branch
    selected_level: ReferenceLevel
    reason_codes: tuple[str, ...]
    evaluations: tuple[ScopeEvaluation, ...]
    config_version: str

    @property
    def can_score(self) -> bool:
        return self.selected_level is not ReferenceLevel.NO_SCORE

    @property
    def fallback_reason(self) -> str | None:
        """Return the first deterministic failure reason, if any."""

        return self.reason_codes[0] if self.reason_codes else None


@dataclass(frozen=True)
class FeaturePersonSupport:
    support_as_of: date | datetime | str
    active_days: int
    span_days: int
    active_days_current_role: int
    coverage: float
    min_feature_observations: int
    last_active_gap_days: int


@dataclass(frozen=True)
class FeatureRoleSupport:
    support_as_of: date | datetime | str
    role_known: bool
    peer_users: int
    peer_user_days: int
    recent_user_days: int
    coverage: float
    min_feature_observations: int


@dataclass(frozen=True)
class FeatureGlobalSupport:
    support_as_of: date | datetime | str
    users: int
    user_days: int
    coverage: float
    train_fitted: bool = True


@dataclass(frozen=True)
class SequencePersonSupport:
    support_as_of: date | datetime | str
    sequence_days: int
    transitions: int
    span_days: int
    sequence_days_current_role: int
    last_active_gap_days: int


@dataclass(frozen=True)
class SequenceRoleSupport:
    support_as_of: date | datetime | str
    role_known: bool
    peer_users: int
    sequence_days: int
    transitions: int
    recent_transitions: int


@dataclass(frozen=True)
class SequenceGlobalSupport:
    support_as_of: date | datetime | str
    users: int
    sequence_days: int
    transitions: int
    train_fitted: bool = True


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _normalize_config_aliases(merged: dict[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    """Map the repository JSON vocabulary to the domain's canonical keys.

    The checked-in framework file intentionally uses descriptive names such
    as ``min_feature_coverage_ratio``.  The domain keeps shorter internal
    names.  Supporting both also makes small test/deployment overrides
    convenient without weakening validation.
    """

    if "version" not in source and "schema_version" in source:
        merged["version"] = source["schema_version"]

    source_readiness = source.get("readiness", {})
    merged_readiness = merged["readiness"]
    aliases: tuple[tuple[str, str, str, str], ...] = (
        (
            "feature",
            "person",
            "min_active_days_in_current_role",
            "min_role_active_days",
        ),
        (
            "feature",
            "person",
            "min_feature_coverage_ratio",
            "min_coverage",
        ),
        (
            "feature",
            "person",
            "min_observations_per_used_feature",
            "min_feature_observations",
        ),
        (
            "feature",
            "role",
            "min_peer_users_excluding_subject",
            "min_peer_users",
        ),
        (
            "feature",
            "role",
            "min_recent_30d_user_days",
            "min_recent_user_days",
        ),
        (
            "feature",
            "role",
            "min_feature_coverage_ratio",
            "min_coverage",
        ),
        (
            "feature",
            "role",
            "min_support_per_feature",
            "min_feature_observations",
        ),
        (
            "feature",
            "global",
            "min_feature_coverage_ratio",
            "min_coverage",
        ),
        (
            "feature",
            "global",
            "fit_train_only",
            "require_train_fitted",
        ),
        (
            "sequence",
            "person",
            "min_sequence_days_in_current_role",
            "min_role_sequence_days",
        ),
        (
            "sequence",
            "person",
            "max_stale_gap_days",
            "max_last_active_gap_days",
        ),
        (
            "sequence",
            "role",
            "min_recent_30d_transitions",
            "min_recent_transitions",
        ),
        (
            "sequence",
            "global",
            "fit_train_only",
            "require_train_fitted",
        ),
    )
    for branch, scope, alias, canonical in aliases:
        source_scope = (
            source_readiness.get(branch, {}).get(scope, {})
            if isinstance(source_readiness, Mapping)
            else {}
        )
        if alias in source_scope and canonical not in source_scope:
            merged_readiness[branch][scope][canonical] = source_scope[alias]

    source_fusion = source.get("fusion", {})
    if isinstance(source_fusion, Mapping):
        weights = source_fusion.get("weights", {})
        if isinstance(weights, Mapping):
            if "feature" in weights and "feature_weight" not in source_fusion:
                merged["fusion"]["feature_weight"] = weights["feature"]
            if "sequence" in weights and "sequence_weight" not in source_fusion:
                merged["fusion"]["sequence_weight"] = weights["sequence"]
    return merged


def load_framework_config(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and deep-merge framework configuration with safe defaults."""

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return deepcopy(DEFAULT_FRAMEWORK_CONFIG)

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load framework config: {config_path}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError("Framework config root must be a JSON object")
    merged = _deep_merge(DEFAULT_FRAMEWORK_CONFIG, payload)
    return _normalize_config_aliases(merged, payload)


def _resolve_config(
    config: Mapping[str, Any] | None,
    config_path: str | Path | None,
) -> dict[str, Any]:
    if config is not None and config_path is not None:
        raise ValueError("Pass config or config_path, not both")
    if config is None:
        return load_framework_config(config_path)
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    merged = _deep_merge(DEFAULT_FRAMEWORK_CONFIG, config)
    return _normalize_config_aliases(merged, config)


def _as_date(value: date | datetime | str, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO date") from exc
    raise TypeError(f"{field_name} must be date, datetime, or ISO date string")


def _support_is_past(
    support_as_of: date | datetime | str,
    score_date: date,
) -> bool:
    return _as_date(support_as_of, "support_as_of") < score_date


def _evaluation(level: ReferenceLevel, reasons: Sequence[str]) -> ScopeEvaluation:
    reason_codes = tuple(reasons)
    return ScopeEvaluation(
        level=level,
        eligible=not reason_codes,
        reason_codes=reason_codes,
    )


def _decision(
    branch: Branch,
    evaluations: list[ScopeEvaluation],
    selected_level: ReferenceLevel,
    terminal_reason: str | None,
    version: str,
) -> ReadinessDecision:
    reasons = [code for evaluation in evaluations for code in evaluation.reason_codes]
    if terminal_reason is not None:
        reasons.append(terminal_reason)
    return ReadinessDecision(
        branch=branch,
        selected_level=selected_level,
        reason_codes=tuple(reasons),
        evaluations=tuple(evaluations),
        config_version=version,
    )


def select_feature_reference(
    *,
    score_date: date | datetime | str,
    person: FeaturePersonSupport | None,
    role: FeatureRoleSupport | None,
    global_support: FeatureGlobalSupport | None,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> ReadinessDecision:
    """Select one Feature reference using strict PERSON -> ROLE -> GLOBAL."""

    resolved = _resolve_config(config, config_path)
    thresholds = resolved["readiness"]["feature"]
    day = _as_date(score_date, "score_date")
    evaluations: list[ScopeEvaluation] = []

    person_reasons: list[str] = []
    if person is None:
        person_reasons.append("P_SUPPORT_MISSING")
    else:
        cfg = thresholds["person"]
        if not _support_is_past(person.support_as_of, day):
            person_reasons.append("P_SUPPORT_NOT_PAST")
        if person.active_days < cfg["min_active_days"]:
            person_reasons.append("P_ACTIVE_DAYS_LOW")
        if person.span_days < cfg["min_span_days"]:
            person_reasons.append("P_SPAN_DAYS_LOW")
        if person.active_days_current_role < cfg["min_role_active_days"]:
            person_reasons.append("P_ROLE_TENURE_LOW")
        if person.coverage < cfg["min_coverage"]:
            person_reasons.append("P_COVERAGE_LOW")
        if person.min_feature_observations < cfg["min_feature_observations"]:
            person_reasons.append("P_FEATURE_SUPPORT_LOW")
        if person.last_active_gap_days > cfg["max_last_active_gap_days"]:
            person_reasons.append("P_STALE")
    person_eval = _evaluation(ReferenceLevel.PERSON, person_reasons)
    evaluations.append(person_eval)
    if person_eval.eligible:
        return _decision(
            Branch.FEATURE,
            evaluations,
            ReferenceLevel.PERSON,
            None,
            str(resolved["version"]),
        )

    role_reasons: list[str] = []
    if role is None:
        role_reasons.append("R_SUPPORT_MISSING")
    else:
        cfg = thresholds["role"]
        if not _support_is_past(role.support_as_of, day):
            role_reasons.append("R_SUPPORT_NOT_PAST")
        if not role.role_known:
            role_reasons.append("R_UNKNOWN")
        if role.peer_users < cfg["min_peer_users"]:
            role_reasons.append("R_USERS_LT_15")
        if role.peer_user_days < cfg["min_peer_user_days"]:
            role_reasons.append("R_USERDAYS_LT_300")
        if role.recent_user_days < cfg["min_recent_user_days"]:
            role_reasons.append("R_RECENT_LOW")
        if role.coverage < cfg["min_coverage"]:
            role_reasons.append("R_COVERAGE_LOW")
        if role.min_feature_observations < cfg["min_feature_observations"]:
            role_reasons.append("R_FEATURE_SUPPORT_LT_200")
    role_eval = _evaluation(ReferenceLevel.ROLE, role_reasons)
    evaluations.append(role_eval)
    if role_eval.eligible:
        return _decision(
            Branch.FEATURE,
            evaluations,
            ReferenceLevel.ROLE,
            None,
            str(resolved["version"]),
        )

    global_reasons: list[str] = []
    if global_support is None:
        global_reasons.append("G_SUPPORT_MISSING")
    else:
        cfg = thresholds["global"]
        if not _support_is_past(global_support.support_as_of, day):
            global_reasons.append("G_SUPPORT_NOT_PAST")
        if global_support.users < cfg["min_users"]:
            global_reasons.append("G_USERS_LT_200")
        if global_support.user_days < cfg["min_user_days"]:
            global_reasons.append("G_USERDAYS_LT_10000")
        if global_support.coverage < cfg["min_coverage"]:
            global_reasons.append("G_COVERAGE_LOW")
        if cfg["require_train_fitted"] and not global_support.train_fitted:
            global_reasons.append("G_NOT_TRAIN_FITTED")
    global_eval = _evaluation(ReferenceLevel.GLOBAL, global_reasons)
    evaluations.append(global_eval)
    if global_eval.eligible:
        return _decision(
            Branch.FEATURE,
            evaluations,
            ReferenceLevel.GLOBAL,
            None,
            str(resolved["version"]),
        )

    return _decision(
        Branch.FEATURE,
        evaluations,
        ReferenceLevel.NO_SCORE,
        "F_INSUFFICIENT_DATA",
        str(resolved["version"]),
    )


def select_sequence_reference(
    *,
    score_date: date | datetime | str,
    seq_len: int,
    person: SequencePersonSupport | None,
    role: SequenceRoleSupport | None,
    global_support: SequenceGlobalSupport | None,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> ReadinessDecision:
    """Select one Sequence reference, independently of Feature readiness."""

    if isinstance(seq_len, bool) or not isinstance(seq_len, int):
        raise TypeError("seq_len must be an integer")
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")

    resolved = _resolve_config(config, config_path)
    thresholds = resolved["readiness"]["sequence"]
    day = _as_date(score_date, "score_date")
    min_seq_len = thresholds["current_day"]["min_seq_len"]
    if seq_len < min_seq_len:
        current_eval = ScopeEvaluation(
            level=ReferenceLevel.NO_SCORE,
            eligible=False,
            reason_codes=("S_CURRENT_LEN_LT_2",),
        )
        return _decision(
            Branch.SEQUENCE,
            [current_eval],
            ReferenceLevel.NO_SCORE,
            None,
            str(resolved["version"]),
        )

    evaluations: list[ScopeEvaluation] = []
    person_reasons: list[str] = []
    if person is None:
        person_reasons.append("SP_SUPPORT_MISSING")
    else:
        cfg = thresholds["person"]
        if not _support_is_past(person.support_as_of, day):
            person_reasons.append("SP_SUPPORT_NOT_PAST")
        if person.sequence_days < cfg["min_sequence_days"]:
            person_reasons.append("SP_DAYS_LOW")
        if person.transitions < cfg["min_transitions"]:
            person_reasons.append("SP_TRANSITIONS_LOW")
        if person.span_days < cfg["min_span_days"]:
            person_reasons.append("SP_SPAN_DAYS_LOW")
        if person.sequence_days_current_role < cfg["min_role_sequence_days"]:
            person_reasons.append("SP_ROLE_TENURE_LOW")
        if person.last_active_gap_days > cfg["max_last_active_gap_days"]:
            person_reasons.append("SP_STALE")
    person_eval = _evaluation(ReferenceLevel.PERSON, person_reasons)
    evaluations.append(person_eval)
    if person_eval.eligible:
        return _decision(
            Branch.SEQUENCE,
            evaluations,
            ReferenceLevel.PERSON,
            None,
            str(resolved["version"]),
        )

    role_reasons: list[str] = []
    if role is None:
        role_reasons.append("SR_SUPPORT_MISSING")
    else:
        cfg = thresholds["role"]
        if not _support_is_past(role.support_as_of, day):
            role_reasons.append("SR_SUPPORT_NOT_PAST")
        if not role.role_known:
            role_reasons.append("SR_UNKNOWN")
        if role.peer_users < cfg["min_peer_users"]:
            role_reasons.append("SR_USERS_LT_15")
        if role.sequence_days < cfg["min_sequence_days"]:
            role_reasons.append("SR_DAYS_LT_300")
        if role.transitions < cfg["min_transitions"]:
            role_reasons.append("SR_TRANS_LT_10000")
        if role.recent_transitions < cfg["min_recent_transitions"]:
            role_reasons.append("SR_RECENT_LOW")
    role_eval = _evaluation(ReferenceLevel.ROLE, role_reasons)
    evaluations.append(role_eval)
    if role_eval.eligible:
        return _decision(
            Branch.SEQUENCE,
            evaluations,
            ReferenceLevel.ROLE,
            None,
            str(resolved["version"]),
        )

    global_reasons: list[str] = []
    if global_support is None:
        global_reasons.append("SG_SUPPORT_MISSING")
    else:
        cfg = thresholds["global"]
        if not _support_is_past(global_support.support_as_of, day):
            global_reasons.append("SG_SUPPORT_NOT_PAST")
        if global_support.users < cfg["min_users"]:
            global_reasons.append("SG_USERS_LT_200")
        if global_support.sequence_days < cfg["min_sequence_days"]:
            global_reasons.append("SG_DAYS_LT_10000")
        if global_support.transitions < cfg["min_transitions"]:
            global_reasons.append("SG_TRANS_LT_100000")
        if cfg["require_train_fitted"] and not global_support.train_fitted:
            global_reasons.append("SG_NOT_TRAIN_FITTED")
    global_eval = _evaluation(ReferenceLevel.GLOBAL, global_reasons)
    evaluations.append(global_eval)
    if global_eval.eligible:
        return _decision(
            Branch.SEQUENCE,
            evaluations,
            ReferenceLevel.GLOBAL,
            None,
            str(resolved["version"]),
        )

    return _decision(
        Branch.SEQUENCE,
        evaluations,
        ReferenceLevel.NO_SCORE,
        "S_INSUFFICIENT_DATA",
        str(resolved["version"]),
    )
