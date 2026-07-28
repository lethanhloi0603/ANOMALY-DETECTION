"""Leakage-resistant preprocessing rules for the metadata-only primary experiment."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from statistics import fmean, pstdev
from urllib.parse import parse_qsl, urlsplit

import numpy as np

from insider_ml.contracts import FEATURE_DIMENSION

_MAD_FACTOR = 1.4826
_SOURCE_RANK = {"LOGON": 0, "DEVICE": 1, "FILE": 2, "HTTP": 3, "EMAIL": 4}


@dataclass(frozen=True, slots=True)
class PCUsage:
    user_id: str
    pc: str
    day: date


@dataclass(frozen=True, slots=True)
class PCProfile:
    pc: str
    distinct_users: int
    dominance: float
    owner_user_id: str | None
    is_shared: bool


@dataclass(frozen=True, slots=True)
class OrderedEvent:
    timestamp: datetime
    source: str
    event_uid: str


@dataclass(frozen=True, slots=True)
class RobustFeatureScaler:
    location: np.ndarray
    scale: np.ndarray
    support: np.ndarray
    checksum: str

    def transform(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        numeric = np.asarray(values, dtype=np.float64)
        observed = np.asarray(mask, dtype=bool)
        if numeric.shape != observed.shape or numeric.shape[-1] != FEATURE_DIMENSION:
            raise ValueError("values/mask must align and end with the locked feature dimension")
        scaled = (numeric - self.location) / self.scale
        return np.where(observed, scaled, 0.0).astype(np.float32)


def fit_pc_profiles(
    usage: Iterable[PCUsage],
    *,
    shared_min_distinct_users: int = 5,
    shared_max_dominance_exclusive: float = 0.5,
    owner_min_dominance_inclusive: float = 0.5,
) -> dict[str, PCProfile]:
    """Fit the frozen Train-only PC map using distinct user-days."""

    if shared_min_distinct_users < 2:
        raise ValueError("shared_min_distinct_users must be at least 2")
    if not 0 < shared_max_dominance_exclusive < 1:
        raise ValueError("shared dominance threshold must be within (0,1)")
    if not 0 < owner_min_dominance_inclusive <= 1:
        raise ValueError("owner dominance threshold must be within (0,1]")

    user_days_by_pc: dict[str, set[tuple[str, date]]] = defaultdict(set)
    for item in usage:
        user_id = item.user_id.strip()
        pc = item.pc.strip()
        if not user_id or not pc:
            continue
        user_days_by_pc[pc].add((user_id, item.day))

    profiles: dict[str, PCProfile] = {}
    for pc, user_days in user_days_by_pc.items():
        counts = Counter(user_id for user_id, _day in user_days)
        total = sum(counts.values())
        highest = max(counts.values())
        dominance = highest / total
        owner_candidates = sorted(user_id for user_id, count in counts.items() if count == highest)
        is_shared = (
            len(counts) >= shared_min_distinct_users
            and dominance < shared_max_dominance_exclusive
        )
        owner = (
            owner_candidates[0]
            if not is_shared and dominance >= owner_min_dominance_inclusive
            else None
        )
        profiles[pc] = PCProfile(
            pc=pc,
            distinct_users=len(counts),
            dominance=dominance,
            owner_user_id=owner,
            is_shared=is_shared,
        )
    return profiles


def classify_pc_context(
    profiles: dict[str, PCProfile],
    *,
    user_id: str,
    pc: str | None,
) -> str:
    if not pc or pc not in profiles:
        return "UNKNOWN"
    profile = profiles[pc]
    if profile.is_shared:
        return "SHARED"
    if profile.owner_user_id is None:
        return "UNKNOWN"
    return "OWN" if profile.owner_user_id == user_id else "FOREIGN"


def normalize_hostname(url: str) -> str | None:
    """Return a deterministic hostname without visiting or resolving the URL."""

    raw = str(url).strip()
    if not raw:
        return None
    parsed = urlsplit(raw if "://" in raw else f"//{raw}")
    hostname = parsed.hostname
    if not hostname:
        return None
    normalized = hostname.rstrip(".").lower()
    try:
        return normalized.encode("idna").decode("ascii")
    except UnicodeError:
        return None


def new_domain_count_30d(
    current_urls: Iterable[str],
    prior_urls: Sequence[str],
    *,
    minimum_prior_http_events: int = 20,
) -> int | None:
    """Count distinct current hosts absent from approved past-only user history."""

    if minimum_prior_http_events < 1:
        raise ValueError("minimum_prior_http_events must be positive")
    if len(prior_urls) < minimum_prior_http_events:
        return None
    history = {host for value in prior_urls if (host := normalize_hostname(value))}
    current = {host for value in current_urls if (host := normalize_hostname(value))}
    return len(current - history)


def file_metadata_features(
    filenames: Sequence[str],
    timestamps: Sequence[datetime],
) -> dict[str, float | None]:
    if len(filenames) != len(timestamps):
        raise ValueError("filenames and timestamps must align")
    if not filenames:
        return {
            "filename_length_mean": None,
            "filename_length_max": None,
            "filename_length_std": None,
            "extension_length_mean": None,
            "extension_length_max": None,
            "extensionless_file_count": 0.0,
            "multi_dot_filename_count": 0.0,
            "distinct_filename_stem_count": 0.0,
            "repeated_extension_copy_count": 0.0,
            "extension_present_ratio": None,
            "file_copy_interarrival_mean_minutes": None,
            "file_copy_interarrival_min_minutes": None,
            "file_copy_burst_count_5m": 0.0,
        }
    basenames = [str(value).replace("\\", "/").rsplit("/", 1)[-1] for value in filenames]
    normalized_extensions: list[str] = []
    stems: list[str] = []
    for basename in basenames:
        head, separator, tail = basename.rpartition(".")
        has_extension = bool(separator and head and tail)
        normalized_extensions.append(tail.casefold() if has_extension else "")
        stems.append((head if has_extension else basename).casefold())
    present_extensions = [value for value in normalized_extensions if value]
    extension_counts = Counter(present_extensions)
    filename_lengths = [len(value) for value in basenames]
    extension_lengths = [len(value) for value in present_extensions]
    ordered = sorted(timestamps)
    gaps = [
        max(0.0, (right - left).total_seconds() / 60.0)
        for left, right in zip(ordered, ordered[1:], strict=False)
    ]
    burst_count = 0
    in_burst = False
    for gap in gaps:
        if gap <= 5.0 and not in_burst:
            burst_count += 1
            in_burst = True
        elif gap > 5.0:
            in_burst = False
    return {
        "filename_length_mean": fmean(filename_lengths),
        "filename_length_max": float(max(filename_lengths)),
        "filename_length_std": pstdev(filename_lengths),
        "extension_length_mean": (
            fmean(extension_lengths) if extension_lengths else None
        ),
        "extension_length_max": (
            float(max(extension_lengths)) if extension_lengths else None
        ),
        "extensionless_file_count": float(
            sum(not value for value in normalized_extensions)
        ),
        "multi_dot_filename_count": float(
            sum(value.count(".") >= 2 for value in basenames)
        ),
        "distinct_filename_stem_count": float(len(set(stems))),
        "repeated_extension_copy_count": float(
            sum(max(count - 1, 0) for count in extension_counts.values())
        ),
        "extension_present_ratio": len(present_extensions) / len(basenames),
        "file_copy_interarrival_mean_minutes": fmean(gaps) if gaps else None,
        "file_copy_interarrival_min_minutes": min(gaps) if gaps else None,
        "file_copy_burst_count_5m": float(burst_count),
    }


def http_metadata_features(urls: Sequence[str]) -> dict[str, float | None]:
    if not urls:
        return {
            "distinct_hostname_count": 0.0,
            "distinct_scheme_count": 0.0,
            "https_request_count": 0.0,
            "query_present_request_count": 0.0,
            "distinct_query_key_count": 0.0,
            "repeated_hostname_request_count": 0.0,
            "hostname_length_mean": None,
            "hostname_length_max": None,
            "url_length_mean": None,
            "url_length_max": None,
            "url_path_depth_mean": None,
            "distinct_url_path_count": 0.0,
        }
    parsed = [urlsplit(value if "://" in value else f"//{value}") for value in urls]
    hostnames = [host for value in urls if (host := normalize_hostname(value))]
    hostname_counts = Counter(hostnames)
    schemes = {item.scheme.lower() for item in parsed if item.scheme}
    query_keys = {
        key.strip().lower()
        for item in parsed
        for key, _value in parse_qsl(item.query, keep_blank_values=True)
        if key.strip()
    }
    depths = [sum(bool(segment) for segment in item.path.split("/")) for item in parsed]
    paths = {item.path or "/" for item in parsed}
    lengths = [len(value) for value in urls]
    return {
        "distinct_hostname_count": float(len(hostname_counts)),
        "distinct_scheme_count": float(len(schemes)),
        "https_request_count": float(sum(item.scheme.lower() == "https" for item in parsed)),
        "query_present_request_count": float(sum(bool(item.query) for item in parsed)),
        "distinct_query_key_count": float(len(query_keys)),
        "repeated_hostname_request_count": float(
            sum(max(count - 1, 0) for count in hostname_counts.values())
        ),
        "hostname_length_mean": fmean(map(len, hostnames)) if hostnames else None,
        "hostname_length_max": float(max(map(len, hostnames))) if hostnames else None,
        "url_length_mean": fmean(lengths),
        "url_length_max": float(max(lengths)),
        "url_path_depth_mean": fmean(depths),
        "distinct_url_path_count": float(len(paths)),
    }


def email_metadata_features(
    recipient_counts: Sequence[int],
    sizes: Sequence[float],
) -> dict[str, float | None]:
    if len(recipient_counts) != len(sizes):
        raise ValueError("recipient counts and sizes must align")
    if not sizes:
        return {"recipient_count_mean": None, "email_size_std": None}
    if any(value < 0 for value in recipient_counts) or any(value < 0 for value in sizes):
        raise ValueError("email metadata values must be non-negative")
    return {
        "recipient_count_mean": fmean(recipient_counts),
        "email_size_std": pstdev(sizes),
    }


def cross_source_transition_features(
    events: Sequence[OrderedEvent],
) -> dict[str, float | None]:
    ordered = sorted(
        events,
        key=lambda item: (
            item.timestamp,
            _SOURCE_RANK.get(item.source.upper(), len(_SOURCE_RANK)),
            item.event_uid,
        ),
    )
    if len(ordered) < 2:
        return {
            "source_transition_count": 0.0,
            "distinct_source_transition_count": 0.0,
            "source_transition_entropy": 0.0,
            "inter_event_gap_p95_minutes": None,
        }
    pairs = [
        (left.source.upper(), right.source.upper())
        for left, right in zip(ordered, ordered[1:], strict=False)
        if left.source.upper() != right.source.upper()
    ]
    counts = Counter(pairs)
    total = sum(counts.values())
    entropy = (
        -sum((count / total) * math.log2(count / total) for count in counts.values())
        if total
        else 0.0
    )
    gaps = sorted(
        max(0.0, (right.timestamp - left.timestamp).total_seconds() / 60.0)
        for left, right in zip(ordered, ordered[1:], strict=False)
    )
    p95_index = max(0, math.ceil(0.95 * len(gaps)) - 1)
    return {
        "source_transition_count": float(total),
        "distinct_source_transition_count": float(len(counts)),
        "source_transition_entropy": entropy,
        "inter_event_gap_p95_minutes": gaps[p95_index],
    }


def fit_robust_feature_scaler(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    minimum_scale: float = 1.0,
) -> RobustFeatureScaler:
    """Fit a median/MAD scaler from Train only; callers own split enforcement."""

    numeric = np.asarray(values, dtype=np.float64)
    observed = np.asarray(mask, dtype=bool)
    if numeric.shape != observed.shape or numeric.shape[-1] != FEATURE_DIMENSION:
        raise ValueError("values/mask must align and end with the locked feature dimension")
    if minimum_scale <= 0:
        raise ValueError("minimum_scale must be positive")

    flattened_values = numeric.reshape(-1, FEATURE_DIMENSION)
    flattened_mask = observed.reshape(-1, FEATURE_DIMENSION)
    location = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    scale = np.full(FEATURE_DIMENSION, minimum_scale, dtype=np.float64)
    support = flattened_mask.sum(axis=0).astype(np.int64)
    for index in range(FEATURE_DIMENSION):
        column = flattened_values[flattened_mask[:, index], index]
        if column.size == 0:
            continue
        center = float(np.median(column))
        mad = float(np.median(np.abs(column - center)))
        location[index] = center
        scale[index] = max(_MAD_FACTOR * mad, minimum_scale)

    payload = {
        "schema_version": "robust-feature-scaler.v1",
        "feature_dimension": FEATURE_DIMENSION,
        "location": location.tolist(),
        "scale": scale.tolist(),
        "support": support.tolist(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return RobustFeatureScaler(
        location=location,
        scale=scale,
        support=support,
        checksum=checksum,
    )
