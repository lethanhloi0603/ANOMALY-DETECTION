"""Past-only temporal baselines without a fixed business-hours assumption."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from math import atan2, cos, fmod, hypot, isfinite, pi, sin
from statistics import fmean, median

MINUTES_PER_DAY = 24.0 * 60.0
_ROBUST_MAD_FACTOR = 1.4826


@dataclass(frozen=True, slots=True)
class TemporalBaseline:
    """Versionable circular-time reference for one hierarchy level."""

    level: str
    scope_key: str
    fitted_through: date
    center_minute: float
    distance_location: float
    distance_scale: float
    observation_count: int


def cyclic_time_components(minute_of_day: float) -> tuple[float, float]:
    """Encode local minute-of-day on the unit circle."""

    minute = float(minute_of_day)
    if not isfinite(minute):
        raise ValueError("minute_of_day must be finite")
    minute = fmod(minute, MINUTES_PER_DAY)
    if minute < 0:
        minute += MINUTES_PER_DAY
    angle = 2.0 * pi * minute / MINUTES_PER_DAY
    return sin(angle), cos(angle)


def circular_distance_minutes(left: float, right: float) -> float:
    """Return the shortest distance between two local clock times."""

    left_value = float(left) % MINUTES_PER_DAY
    right_value = float(right) % MINUTES_PER_DAY
    distance = abs(left_value - right_value)
    return float(min(distance, MINUTES_PER_DAY - distance))


def fit_temporal_baseline(
    minutes: Iterable[float],
    *,
    level: str,
    scope_key: str,
    fitted_through: date,
    minimum_scale_minutes: float = 1.0,
) -> TemporalBaseline:
    """Fit a robust circular reference using approved history only."""

    values = [float(value) for value in minutes]
    if not values:
        raise ValueError("cannot fit a temporal baseline without observations")
    if not all(isfinite(value) for value in values):
        raise ValueError("temporal baseline observations must be finite")
    if minimum_scale_minutes <= 0:
        raise ValueError("minimum_scale_minutes must be positive")

    normalized = [value % MINUTES_PER_DAY for value in values]
    angles = [2.0 * pi * value / MINUTES_PER_DAY for value in normalized]
    mean_sin = fmean(sin(angle) for angle in angles)
    mean_cos = fmean(cos(angle) for angle in angles)
    if hypot(mean_sin, mean_cos) < 1e-8:
        raise ValueError("temporal observations do not have a stable circular center")

    center_angle = atan2(mean_sin, mean_cos) % (2.0 * pi)
    center_minute = center_angle * MINUTES_PER_DAY / (2.0 * pi)
    distances = [
        circular_distance_minutes(value, center_minute) for value in normalized
    ]
    distance_location = float(median(distances))
    mad = float(median(abs(distance - distance_location) for distance in distances))
    distance_scale = max(_ROBUST_MAD_FACTOR * mad, float(minimum_scale_minutes))

    return TemporalBaseline(
        level=level.upper(),
        scope_key=scope_key,
        fitted_through=fitted_through,
        center_minute=center_minute,
        distance_location=distance_location,
        distance_scale=distance_scale,
        observation_count=len(values),
    )


def select_temporal_baseline(
    event_day: date,
    *,
    person: TemporalBaseline | None = None,
    role: TemporalBaseline | None = None,
    global_: TemporalBaseline | None = None,
) -> TemporalBaseline:
    """Select the first eligible past-only reference in Person -> Role -> Global order."""

    expected_levels = (
        ("PERSON", person),
        ("ROLE", role),
        ("GLOBAL", global_),
    )
    for expected_level, baseline in expected_levels:
        if baseline is None:
            continue
        if baseline.level != expected_level:
            raise ValueError(
                f"{expected_level.lower()} reference has level={baseline.level!r}"
            )
        if baseline.fitted_through < event_day:
            return baseline
    raise ValueError("no past-only temporal baseline is available")


def unusual_time_relative_to_baseline(
    minute_of_day: float,
    baseline: TemporalBaseline,
) -> float:
    """Return a non-negative robust z-score of circular-time deviation."""

    distance = circular_distance_minutes(minute_of_day, baseline.center_minute)
    return max(
        0.0,
        (distance - baseline.distance_location) / baseline.distance_scale,
    )


def summarize_temporal_deviation(
    minutes: Iterable[float],
    baseline: TemporalBaseline,
) -> tuple[float, float]:
    """Return mean/max event deviation for one user-day feature row."""

    scores = [
        unusual_time_relative_to_baseline(value, baseline) for value in minutes
    ]
    if not scores:
        raise ValueError("cannot summarize temporal deviation without events")
    return fmean(scores), max(scores)


def build_temporal_feature_values(
    event_minutes: Mapping[str, Iterable[float]],
    baselines: Mapping[str, TemporalBaseline],
) -> dict[str, float | None]:
    """Build the nine feature128.v5 time fields from selected references.

    The caller selects each reference with :func:`select_temporal_baseline`.
    Missing event groups remain missing rather than being silently converted
    to zero.
    """

    groups = {
        name.upper(): [float(value) for value in values]
        for name, values in event_minutes.items()
    }
    references = {name.upper(): value for name, value in baselines.items()}

    def scores(name: str) -> list[float]:
        values = groups.get(name, [])
        if not values:
            return []
        baseline = references.get(name)
        if baseline is None:
            raise ValueError(f"missing selected temporal baseline for {name}")
        return [
            unusual_time_relative_to_baseline(value, baseline) for value in values
        ]

    first_logon = scores("FIRST_LOGON")
    last_logoff = scores("LAST_LOGOFF")
    session_events = scores("LOGON_SESSION")
    device = scores("DEVICE")
    file_events = scores("FILE")
    http = scores("HTTP")
    email = scores("EMAIL")
    all_events = scores("ALL")

    return {
        "first_logon_time_deviation": first_logon[0] if first_logon else None,
        "last_logoff_time_deviation": last_logoff[-1] if last_logoff else None,
        "session_time_deviation_max": max(session_events) if session_events else None,
        "device_time_deviation_mean": fmean(device) if device else None,
        "file_time_deviation_mean": fmean(file_events) if file_events else None,
        "http_time_deviation_mean": fmean(http) if http else None,
        "email_time_deviation_mean": fmean(email) if email else None,
        "unusual_time_relative_to_baseline_mean": (
            fmean(all_events) if all_events else None
        ),
        "unusual_time_relative_to_baseline_max": (
            max(all_events) if all_events else None
        ),
    }
