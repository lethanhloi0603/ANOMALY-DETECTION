"""CERT r4.2 metadata-only preparation for the locked research contracts.

The module deliberately reads the raw CSV files in place.  It never copies the
``content`` columns into an artifact.  Preparation is user-shard aware so a
large CERT source can be processed without retaining every employee in memory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import fmean, pstdev
from urllib.parse import urlsplit

import numpy as np

from insider_ml.contracts import (
    FEATURE_DIMENSION,
    FEATURE_SCHEMA_VERSION,
    MAX_SEQUENCE_LENGTH,
    SEQUENCE_SCHEMA_VERSION,
    SEQUENCE_TOKENS,
    WINDOW_DAYS,
    PreparedWindows,
    locked_split_for_day,
)
from insider_ml.preprocessing import (
    OrderedEvent,
    PCProfile,
    PCUsage,
    RobustFeatureScaler,
    classify_pc_context,
    cross_source_transition_features,
    email_metadata_features,
    file_metadata_features,
    fit_pc_profiles,
    fit_robust_feature_scaler,
    http_metadata_features,
    normalize_hostname,
)
from insider_ml.temporal import (
    TemporalBaseline,
    cyclic_time_components,
    fit_temporal_baseline,
    unusual_time_relative_to_baseline,
)

CERT_TIME_FORMAT = "%m/%d/%Y %H:%M:%S"
SOURCE_FILES = {
    "LOGON": "logon.csv",
    "DEVICE": "device.csv",
    "FILE": "file.csv",
    "HTTP": "http.csv",
    "EMAIL": "email.csv",
}
SOURCE_RANK = {name: rank for rank, name in enumerate(SOURCE_FILES)}
TOKEN_IDS = {name: index for index, name in enumerate(SEQUENCE_TOKENS)}
PC_IDS = {"OWN": 1, "SHARED": 2, "FOREIGN": 3, "UNKNOWN": 4}
CALENDAR_IDS = {"WEEKDAY": 1, "WEEKEND": 2}
COUNT_FEATURES = {
    "logon_count",
    "logoff_count",
    "weekend_logon_count",
    "distinct_logon_pc_count",
    "own_pc_logon_count",
    "shared_pc_logon_count",
    "foreign_pc_logon_count",
    "matched_logon_session_count",
    "device_connect_count",
    "device_disconnect_count",
    "device_session_count",
    "unmatched_connect_count",
    "unmatched_disconnect_count",
    "weekend_device_connect_count",
    "own_pc_device_connect_count",
    "shared_pc_device_connect_count",
    "foreign_pc_device_connect_count",
    "file_copy_count",
    "distinct_filename_count",
    "distinct_extension_count",
    "extensionless_file_count",
    "multi_dot_filename_count",
    "distinct_filename_stem_count",
    "repeated_extension_copy_count",
    "weekend_file_count",
    "own_pc_file_count",
    "shared_pc_file_count",
    "foreign_pc_file_count",
    "distinct_file_pc_count",
    "repeated_filename_copy_count",
    "file_copy_burst_count_5m",
    "http_request_count",
    "distinct_url_count",
    "distinct_domain_count",
    "distinct_path_count",
    "weekend_http_count",
    "own_pc_http_count",
    "shared_pc_http_count",
    "foreign_pc_http_count",
    "distinct_hostname_count",
    "distinct_scheme_count",
    "https_request_count",
    "query_present_request_count",
    "distinct_query_key_count",
    "repeated_hostname_request_count",
    "distinct_url_path_count",
    "repeated_url_request_count",
    "email_sent_count",
    "weekend_email_count",
    "own_pc_email_count",
    "shared_pc_email_count",
    "foreign_pc_email_count",
    "recipient_total_count",
    "unique_recipient_count",
    "internal_recipient_total",
    "external_recipient_total",
    "external_recipient_email_count",
    "to_recipient_total",
    "cc_recipient_total",
    "bcc_recipient_total",
    "bcc_email_count",
    "attachment_email_count",
    "attachment_total_count",
    "attachment_max_count",
    "personal_from_email_count",
    "new_external_recipient_count_30d",
    "total_event_count",
    "active_channel_count",
    "weekend_event_count",
    "own_pc_event_count",
    "shared_pc_event_count",
    "foreign_pc_event_count",
    "distinct_all_pc_count",
    "source_transition_count",
    "distinct_source_transition_count",
    "multi_channel_burst_count",
    "active_days_last_7",
    "active_days_last_30",
    "days_in_current_role",
}
PRIMARY_DISABLED_FEATURES = {
    "source_transition_count",
    "distinct_source_transition_count",
    "source_transition_entropy",
    "inter_event_gap_p95_minutes",
}


@dataclass(frozen=True, slots=True)
class LdapProfile:
    user_id: str
    role: str
    department: str
    team: str
    email: str


@dataclass(frozen=True, slots=True)
class CertEvent:
    event_uid: str
    timestamp: datetime
    user_id: str
    pc: str
    source: str
    action: str
    metadata: dict[str, str]

    @property
    def day(self) -> date:
        return self.timestamp.date()

    @property
    def minute(self) -> float:
        return self.timestamp.hour * 60.0 + self.timestamp.minute + self.timestamp.second / 60.0


@dataclass(slots=True)
class PreparedDay:
    values: np.ndarray
    mask: np.ndarray
    sequence: list[CertEvent]
    role: str


class LdapDirectory:
    """Monthly LDAP snapshots with first-of-month effective dates."""

    def __init__(self, snapshots: Sequence[tuple[date, dict[str, LdapProfile]]]) -> None:
        if not snapshots:
            raise ValueError("LDAP directory has no snapshots")
        self.snapshots = tuple(sorted(snapshots, key=lambda item: item[0]))

    @classmethod
    def from_raw(cls, raw_root: Path) -> LdapDirectory:
        snapshots: list[tuple[date, dict[str, LdapProfile]]] = []
        for path in sorted((raw_root / "LDAP").glob("????-??.csv")):
            effective = date.fromisoformat(f"{path.stem}-01")
            profiles: dict[str, LdapProfile] = {}
            with path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    user_id = str(row.get("user_id", "")).strip()
                    if not user_id:
                        continue
                    profiles[user_id] = LdapProfile(
                        user_id=user_id,
                        role=str(row.get("role", "")).strip() or "UNKNOWN_ROLE",
                        department=str(row.get("department", "")).strip(),
                        team=str(row.get("team", "")).strip(),
                        email=str(row.get("email", "")).strip().lower(),
                    )
            snapshots.append((effective, profiles))
        return cls(snapshots)

    def _snapshot_index(self, day: date) -> int | None:
        selected: int | None = None
        for index, (effective, _profiles) in enumerate(self.snapshots):
            if effective > day:
                break
            selected = index
        return selected

    def profile(self, user_id: str, day: date) -> LdapProfile | None:
        index = self._snapshot_index(day)
        return None if index is None else self.snapshots[index][1].get(user_id)

    def users_on(self, day: date) -> set[str]:
        index = self._snapshot_index(day)
        return set() if index is None else set(self.snapshots[index][1])

    def role_start(self, user_id: str, day: date) -> date | None:
        index = self._snapshot_index(day)
        if index is None:
            return None
        profile = self.snapshots[index][1].get(user_id)
        if profile is None:
            return None
        start = self.snapshots[index][0]
        for prior in range(index - 1, -1, -1):
            prior_profile = self.snapshots[prior][1].get(user_id)
            if prior_profile is None or prior_profile.role != profile.role:
                break
            start = self.snapshots[prior][0]
        return start


def load_feature_names(config_path: Path) -> tuple[str, ...]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"unsupported feature catalog: {config_path}")
    names = tuple(str(item["name"]) for item in payload["features"])
    if len(names) != FEATURE_DIMENSION or len(set(names)) != FEATURE_DIMENSION:
        raise ValueError("feature catalog must contain 128 unique feature names")
    return names


def select_users(
    raw_root: Path,
    *,
    start_day: date,
    end_day: date,
    max_users: int,
) -> tuple[str, ...]:
    if max_users < 1:
        raise ValueError("max_users must be positive; run CERT in explicit user shards")
    selected: set[str] = set()
    with (raw_root / "logon.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            event_day = datetime.strptime(row["date"], CERT_TIME_FORMAT).date()
            if event_day < start_day:
                continue
            if event_day > end_day:
                break
            selected.add(row["user"].strip())
            if len(selected) >= max_users:
                break
    if not selected:
        raise ValueError("no users were found in the selected raw date range")
    return tuple(sorted(selected))


def _metadata_for_source(source: str, row: dict[str, str]) -> dict[str, str]:
    if source == "FILE":
        return {"filename": str(row.get("filename", ""))}
    if source == "HTTP":
        return {"url": str(row.get("url", ""))}
    if source == "EMAIL":
        return {
            key: str(row.get(key, ""))
            for key in ("to", "cc", "bcc", "from", "size", "attachments")
        }
    return {}


def load_events(
    raw_root: Path,
    users: Iterable[str],
    *,
    start_day: date,
    end_day: date,
    max_rows_per_source: int | None = None,
) -> tuple[list[CertEvent], dict[str, int]]:
    """Load a bounded user shard and drop all raw content immediately."""

    selected = set(users)
    events: list[CertEvent] = []
    rows_read: dict[str, int] = {}
    for source, filename in SOURCE_FILES.items():
        count = 0
        with (raw_root / filename).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if max_rows_per_source is not None and count >= max_rows_per_source:
                    break
                count += 1
                timestamp = datetime.strptime(row["date"], CERT_TIME_FORMAT)
                if timestamp.date() < start_day:
                    continue
                if timestamp.date() > end_day:
                    break
                user_id = str(row.get("user", "")).strip()
                if user_id not in selected:
                    continue
                events.append(
                    CertEvent(
                        event_uid=str(row.get("id", "")).strip(),
                        timestamp=timestamp,
                        user_id=user_id,
                        pc=str(row.get("pc", "")).strip(),
                        source=source,
                        action=str(row.get("activity", "")).strip().upper(),
                        metadata=_metadata_for_source(source, row),
                    )
                )
        rows_read[source] = count
    events.sort(
        key=lambda item: (
            item.timestamp,
            SOURCE_RANK[item.source],
            item.event_uid,
        )
    )
    return events, rows_read


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _entropy(values: Iterable[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _recipients(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(";") if item.strip()]


def _pc_counts(
    events: Sequence[CertEvent],
    profiles: dict[str, PCProfile],
    user_id: str,
) -> Counter[str]:
    return Counter(
        classify_pc_context(profiles, user_id=user_id, pc=event.pc) for event in events
    )


def _paired_minutes(
    events: Sequence[CertEvent],
    *,
    start_action: str,
    end_action: str,
) -> tuple[list[float], int, int]:
    open_by_pc: dict[str, deque[datetime]] = defaultdict(deque)
    durations: list[float] = []
    unmatched_end = 0
    for event in sorted(events, key=lambda item: item.timestamp):
        if event.action == start_action:
            open_by_pc[event.pc].append(event.timestamp)
        elif event.action == end_action:
            if not open_by_pc[event.pc]:
                unmatched_end += 1
                continue
            start = open_by_pc[event.pc].popleft()
            durations.append(max(0.0, (event.timestamp - start).total_seconds() / 60.0))
    unmatched_start = sum(len(queue) for queue in open_by_pc.values())
    return durations, unmatched_start, unmatched_end


def _gap_bucket(minutes: float) -> int:
    if minutes < 1.0:
        return 1
    if minutes < 5.0:
        return 2
    if minutes < 30.0:
        return 3
    if minutes <= 120.0:
        return 4
    return 5


def _event_token(event: CertEvent) -> int:
    if event.source == "LOGON":
        return TOKEN_IDS["LOGOFF" if event.action == "LOGOFF" else "LOGON"]
    if event.source == "DEVICE":
        return TOKEN_IDS[
            "DEVICE_DISCONNECT" if event.action == "DISCONNECT" else "DEVICE_CONNECT"
        ]
    return TOKEN_IDS[event.source]


def _head_tail(events: Sequence[CertEvent]) -> list[CertEvent]:
    if len(events) <= MAX_SEQUENCE_LENGTH:
        return list(events)
    return [*events[:128], *events[-128:]]


def _multi_channel_bursts(events: Sequence[CertEvent]) -> float:
    if len(events) < 2:
        return 0.0
    bursts = 0
    run_sources = {events[0].source}
    in_run = False
    for left, right in zip(events, events[1:], strict=False):
        gap = (right.timestamp - left.timestamp).total_seconds() / 60.0
        if gap <= 5.0:
            run_sources.add(right.source)
            in_run = True
        else:
            if in_run and len(run_sources) >= 2:
                bursts += 1
            run_sources = {right.source}
            in_run = False
    if in_run and len(run_sources) >= 2:
        bursts += 1
    return float(bursts)


def _selected_temporal_baseline(
    *,
    day: date,
    user_id: str,
    role: str,
    person_minutes: dict[str, list[float]],
    person_days: dict[str, set[date]],
    role_minutes: dict[str, list[float]],
    role_days: dict[str, set[tuple[str, date]]],
    global_minutes: list[float],
    global_days: set[tuple[str, date]],
) -> TemporalBaseline | None:
    person_key = f"{user_id}|{role}"
    candidates: list[tuple[str, str, list[float]]] = []
    if len(person_days[person_key]) >= 30:
        candidates.append(("PERSON", person_key, person_minutes[person_key]))
    role_support = role_days[role]
    if len({item[0] for item in role_support if item[0] != user_id}) >= 15 and len(
        role_support
    ) >= 300:
        candidates.append(("ROLE", role, role_minutes[role]))
    if len({item[0] for item in global_days}) >= 200 and len(global_days) >= 10000:
        candidates.append(("GLOBAL", "GLOBAL", global_minutes))
    for level, key, minutes in candidates:
        try:
            return fit_temporal_baseline(
                minutes,
                level=level,
                scope_key=key,
                fitted_through=day - timedelta(days=1),
            )
        except ValueError:
            continue
    return None


def _feature_row(
    *,
    user_id: str,
    day: date,
    role: str,
    role_start: date,
    events: Sequence[CertEvent],
    feature_names: Sequence[str],
    pc_profiles: dict[str, PCProfile],
    active_history: Sequence[tuple[date, int]],
    domain_history: Sequence[tuple[date, set[str], int]],
    recipient_history: Sequence[tuple[date, set[str]]],
    temporal_baseline: TemporalBaseline | None,
) -> tuple[np.ndarray, np.ndarray]:
    values: dict[str, float | None] = {name: None for name in feature_names}
    for name in COUNT_FEATURES:
        if name in values:
            values[name] = 0.0
    by_source = {
        source: [event for event in events if event.source == source]
        for source in SOURCE_FILES
    }
    weekend = day.weekday() >= 5

    logons = by_source["LOGON"]
    logon_only = [event for event in logons if event.action == "LOGON"]
    logoff_only = [event for event in logons if event.action == "LOGOFF"]
    logon_pc = _pc_counts(logon_only, pc_profiles, user_id)
    logon_durations, _unmatched_logon, _unmatched_logoff = _paired_minutes(
        logons,
        start_action="LOGON",
        end_action="LOGOFF",
    )
    first_logon = min((event.minute for event in logon_only), default=None)
    last_logon = max((event.minute for event in logon_only), default=None)
    last_logoff = max((event.minute for event in logoff_only), default=None)
    values.update(
        {
            "logon_count": float(len(logon_only)),
            "logoff_count": float(len(logoff_only)),
            "weekend_logon_count": float(len(logon_only) if weekend else 0),
            "distinct_logon_pc_count": float(len({event.pc for event in logon_only})),
            "own_pc_logon_count": float(logon_pc["OWN"]),
            "shared_pc_logon_count": float(logon_pc["SHARED"]),
            "foreign_pc_logon_count": float(logon_pc["FOREIGN"]),
            "foreign_pc_logon_ratio": _ratio(logon_pc["FOREIGN"], len(logon_only)),
            "first_logon_minute": first_logon,
            "last_logon_minute": last_logon,
            "last_logoff_minute": last_logoff,
            "work_span_minutes": (
                max(0.0, (last_logoff or last_logon) - first_logon)
                if first_logon is not None and (last_logoff is not None or last_logon is not None)
                else None
            ),
            "matched_logon_session_count": float(len(logon_durations)),
            "total_logon_session_minutes": sum(logon_durations) if logon_durations else None,
            "max_logon_session_minutes": max(logon_durations) if logon_durations else None,
        }
    )

    devices = by_source["DEVICE"]
    connects = [event for event in devices if event.action == "CONNECT"]
    disconnects = [event for event in devices if event.action == "DISCONNECT"]
    device_pc = _pc_counts(connects, pc_profiles, user_id)
    device_durations, unmatched_connect, unmatched_disconnect = _paired_minutes(
        devices,
        start_action="CONNECT",
        end_action="DISCONNECT",
    )
    values.update(
        {
            "device_connect_count": float(len(connects)),
            "device_disconnect_count": float(len(disconnects)),
            "device_session_count": float(len(device_durations)),
            "unmatched_connect_count": float(unmatched_connect),
            "unmatched_disconnect_count": float(unmatched_disconnect),
            "weekend_device_connect_count": float(len(connects) if weekend else 0),
            "own_pc_device_connect_count": float(device_pc["OWN"]),
            "shared_pc_device_connect_count": float(device_pc["SHARED"]),
            "foreign_pc_device_connect_count": float(device_pc["FOREIGN"]),
            "device_active_minutes": sum(device_durations) if device_durations else None,
            "max_device_session_minutes": max(device_durations) if device_durations else None,
        }
    )

    files = by_source["FILE"]
    filenames = [event.metadata["filename"] for event in files]
    extensions = [
        Path(name.replace("\\", "/")).suffix.casefold().lstrip(".")
        for name in filenames
    ]
    file_pc = _pc_counts(files, pc_profiles, user_id)
    values.update(file_metadata_features(filenames, [event.timestamp for event in files]))
    values.update(
        {
            "file_copy_count": float(len(files)),
            "distinct_filename_count": float(len(set(map(str.casefold, filenames)))),
            "distinct_extension_count": float(len({value for value in extensions if value})),
            "weekend_file_count": float(len(files) if weekend else 0),
            "own_pc_file_count": float(file_pc["OWN"]),
            "shared_pc_file_count": float(file_pc["SHARED"]),
            "foreign_pc_file_count": float(file_pc["FOREIGN"]),
            "distinct_file_pc_count": float(len({event.pc for event in files})),
            "repeated_filename_copy_count": float(
                sum(max(count - 1, 0) for count in Counter(map(str.casefold, filenames)).values())
            ),
            "file_extension_entropy": _entropy(value for value in extensions if value),
        }
    )

    http = by_source["HTTP"]
    urls = [event.metadata["url"] for event in http]
    hosts = [host for url in urls if (host := normalize_hostname(url))]
    parsed_urls = [urlsplit(url if "://" in url else f"//{url}") for url in urls]
    http_pc = _pc_counts(http, pc_profiles, user_id)
    prior_domains = {
        host
        for history_day, history_hosts, _count in domain_history
        if 0 < (day - history_day).days <= 30
        for host in history_hosts
    }
    prior_http_count = sum(
        count
        for history_day, _history_hosts, count in domain_history
        if 0 < (day - history_day).days <= 30
    )
    values.update(http_metadata_features(urls))
    values.update(
        {
            "http_request_count": float(len(http)),
            "distinct_url_count": float(len(set(urls))),
            "distinct_domain_count": float(len(set(hosts))),
            "distinct_path_count": float(len({item.path or "/" for item in parsed_urls})),
            "weekend_http_count": float(len(http) if weekend else 0),
            "own_pc_http_count": float(http_pc["OWN"]),
            "shared_pc_http_count": float(http_pc["SHARED"]),
            "foreign_pc_http_count": float(http_pc["FOREIGN"]),
            "domain_entropy": _entropy(hosts),
            "repeated_url_request_count": float(
                sum(max(count - 1, 0) for count in Counter(urls).values())
            ),
            "new_domain_count_30d": (
                float(len(set(hosts) - prior_domains)) if prior_http_count >= 20 else None
            ),
        }
    )

    emails = by_source["EMAIL"]
    email_pc = _pc_counts(emails, pc_profiles, user_id)
    recipient_lists = [
        {
            "to": _recipients(event.metadata["to"]),
            "cc": _recipients(event.metadata["cc"]),
            "bcc": _recipients(event.metadata["bcc"]),
        }
        for event in emails
    ]
    all_recipients = [
        recipient
        for groups in recipient_lists
        for values_for_group in groups.values()
        for recipient in values_for_group
    ]
    external = [item for item in all_recipients if not item.endswith("@dtaa.com")]
    internal = [item for item in all_recipients if item.endswith("@dtaa.com")]
    sizes = [max(0.0, float(event.metadata["size"] or 0)) for event in emails]
    attachments = [max(0, int(event.metadata["attachments"] or 0)) for event in emails]
    recipient_counts = [sum(len(group) for group in groups.values()) for groups in recipient_lists]
    prior_external = {
        recipient
        for history_day, history_recipients in recipient_history
        if 0 < (day - history_day).days <= 30
        for recipient in history_recipients
    }
    values.update(email_metadata_features(recipient_counts, sizes))
    values.update(
        {
            "email_sent_count": float(len(emails)),
            "weekend_email_count": float(len(emails) if weekend else 0),
            "own_pc_email_count": float(email_pc["OWN"]),
            "shared_pc_email_count": float(email_pc["SHARED"]),
            "foreign_pc_email_count": float(email_pc["FOREIGN"]),
            "recipient_total_count": float(len(all_recipients)),
            "unique_recipient_count": float(len(set(all_recipients))),
            "internal_recipient_total": float(len(internal)),
            "external_recipient_total": float(len(external)),
            "external_recipient_email_count": float(
                sum(
                    any(
                        not item.endswith("@dtaa.com")
                        for group in groups.values()
                        for item in group
                    )
                    for groups in recipient_lists
                )
            ),
            "external_recipient_ratio": _ratio(len(external), len(all_recipients)),
            "to_recipient_total": float(sum(len(groups["to"]) for groups in recipient_lists)),
            "cc_recipient_total": float(sum(len(groups["cc"]) for groups in recipient_lists)),
            "bcc_recipient_total": float(sum(len(groups["bcc"]) for groups in recipient_lists)),
            "bcc_email_count": float(sum(bool(groups["bcc"]) for groups in recipient_lists)),
            "attachment_email_count": float(sum(value > 0 for value in attachments)),
            "attachment_total_count": float(sum(attachments)),
            "attachment_max_count": float(max(attachments, default=0)),
            "email_size_total": sum(sizes) if emails else None,
            "email_size_mean": fmean(sizes) if emails else None,
            "email_size_max": max(sizes) if emails else None,
            "personal_from_email_count": float(
                sum(
                    bool(event.metadata["from"])
                    and not event.metadata["from"].strip().lower().endswith("@dtaa.com")
                    for event in emails
                )
            ),
            "new_external_recipient_count_30d": (
                float(len(set(external) - prior_external))
                if sum(
                    len(history_recipients)
                    for history_day, history_recipients in recipient_history
                    if 0 < (day - history_day).days <= 30
                )
                >= 5
                else None
            ),
        }
    )

    all_pc = _pc_counts(events, pc_profiles, user_id)
    transitions = cross_source_transition_features(
        [
            OrderedEvent(
                timestamp=event.timestamp,
                source=event.source,
                event_uid=event.event_uid,
            )
            for event in events
        ]
    )
    values.update(transitions)
    values.update(
        {
            "total_event_count": float(len(events)),
            "active_channel_count": float(sum(bool(items) for items in by_source.values())),
            "weekend_event_count": float(len(events) if weekend else 0),
            "weekend_event_ratio": _ratio(len(events) if weekend else 0, len(events)),
            "own_pc_event_count": float(all_pc["OWN"]),
            "shared_pc_event_count": float(all_pc["SHARED"]),
            "foreign_pc_event_count": float(all_pc["FOREIGN"]),
            "foreign_pc_event_ratio": _ratio(all_pc["FOREIGN"], len(events)),
            "distinct_all_pc_count": float(len({event.pc for event in events})),
            "multi_channel_burst_count": _multi_channel_bursts(events),
        }
    )

    for horizon in (7, 30):
        history = [
            count
            for history_day, count in active_history
            if 0 < (day - history_day).days <= horizon
        ]
        if horizon == 7:
            values["active_days_last_7"] = float(sum(count > 0 for count in history))
            values["total_events_mean_7d"] = fmean(history) if history else 0.0
        else:
            values["active_days_last_30"] = float(sum(count > 0 for count in history))
            values["total_events_mean_30d"] = fmean(history) if history else 0.0
            values["total_events_std_30d"] = pstdev(history) if history else 0.0
    values["total_events_short_long_ratio"] = _ratio(
        float(values["total_events_mean_7d"] or 0),
        float(values["total_events_mean_30d"] or 0),
    )
    prior_active = [history_day for history_day, count in active_history if count > 0]
    values["days_since_last_active_day"] = (
        float((day - max(prior_active)).days) if prior_active else None
    )
    values["days_in_current_role"] = float((day - role_start).days + 1)

    if temporal_baseline is not None and events:
        deviations = [
            unusual_time_relative_to_baseline(event.minute, temporal_baseline)
            for event in events
        ]
        values["unusual_time_relative_to_baseline_mean"] = fmean(deviations)
        values["unusual_time_relative_to_baseline_max"] = max(deviations)
        source_feature = {
            "DEVICE": "device_time_deviation_mean",
            "FILE": "file_time_deviation_mean",
            "HTTP": "http_time_deviation_mean",
            "EMAIL": "email_time_deviation_mean",
        }
        for source, feature_name in source_feature.items():
            source_values = [
                unusual_time_relative_to_baseline(event.minute, temporal_baseline)
                for event in by_source[source]
            ]
            values[feature_name] = fmean(source_values) if source_values else None
        if logon_only:
            values["first_logon_time_deviation"] = unusual_time_relative_to_baseline(
                min(logon_only, key=lambda item: item.timestamp).minute,
                temporal_baseline,
            )
        if logoff_only:
            values["last_logoff_time_deviation"] = unusual_time_relative_to_baseline(
                max(logoff_only, key=lambda item: item.timestamp).minute,
                temporal_baseline,
            )
        session_boundaries = [*logon_only, *logoff_only]
        if session_boundaries:
            values["session_time_deviation_max"] = max(
                unusual_time_relative_to_baseline(event.minute, temporal_baseline)
                for event in session_boundaries
            )

    vector = np.zeros(FEATURE_DIMENSION, dtype=np.float32)
    mask = np.zeros(FEATURE_DIMENSION, dtype=bool)
    for index, name in enumerate(feature_names):
        if name in PRIMARY_DISABLED_FEATURES:
            continue
        value = values[name]
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        vector[index] = numeric
        mask[index] = True
    return vector, mask


def build_prepared_days(
    *,
    events: Sequence[CertEvent],
    users: Sequence[str],
    directory: LdapDirectory,
    feature_names: Sequence[str],
    start_day: date,
    end_day: date,
) -> tuple[dict[tuple[str, date], PreparedDay], dict[str, PCProfile]]:
    usage = [
        PCUsage(user_id=event.user_id, pc=event.pc, day=event.day)
        for event in events
        if locked_split_for_day(event.day) == "TRAIN"
    ]
    pc_profiles = fit_pc_profiles(usage)
    grouped: dict[tuple[str, date], list[CertEvent]] = defaultdict(list)
    for event in events:
        grouped[(event.user_id, event.day)].append(event)

    prepared: dict[tuple[str, date], PreparedDay] = {}
    active_history: dict[str, list[tuple[date, int]]] = defaultdict(list)
    domain_history: dict[str, list[tuple[date, set[str], int]]] = defaultdict(list)
    recipient_history: dict[str, list[tuple[date, set[str]]]] = defaultdict(list)
    person_minutes: dict[str, list[float]] = defaultdict(list)
    person_days: dict[str, set[date]] = defaultdict(set)
    role_minutes: dict[str, list[float]] = defaultdict(list)
    role_days: dict[str, set[tuple[str, date]]] = defaultdict(set)
    global_minutes: list[float] = []
    global_days: set[tuple[str, date]] = set()

    current = start_day
    while current <= end_day:
        for user_id in users:
            profile = directory.profile(user_id, current)
            role_start = directory.role_start(user_id, current)
            if profile is None or role_start is None:
                continue
            day_events = grouped.get((user_id, current), [])
            baseline = _selected_temporal_baseline(
                day=current,
                user_id=user_id,
                role=profile.role,
                person_minutes=person_minutes,
                person_days=person_days,
                role_minutes=role_minutes,
                role_days=role_days,
                global_minutes=global_minutes,
                global_days=global_days,
            )
            values, mask = _feature_row(
                user_id=user_id,
                day=current,
                role=profile.role,
                role_start=role_start,
                events=day_events,
                feature_names=feature_names,
                pc_profiles=pc_profiles,
                active_history=active_history[user_id],
                domain_history=domain_history[user_id],
                recipient_history=recipient_history[user_id],
                temporal_baseline=baseline,
            )
            prepared[(user_id, current)] = PreparedDay(
                values=values,
                mask=mask,
                sequence=_head_tail(day_events),
                role=profile.role,
            )
            active_history[user_id].append((current, len(day_events)))
            http_events = [event for event in day_events if event.source == "HTTP"]
            domain_history[user_id].append(
                (
                    current,
                    {
                        host
                        for event in http_events
                        if (host := normalize_hostname(event.metadata["url"]))
                    },
                    len(http_events),
                )
            )
            external_recipients = {
                recipient
                for event in day_events
                if event.source == "EMAIL"
                for field in ("to", "cc", "bcc")
                for recipient in _recipients(event.metadata[field])
                if not recipient.endswith("@dtaa.com")
            }
            recipient_history[user_id].append((current, external_recipients))
            if locked_split_for_day(current) == "TRAIN" and day_events:
                minutes = [event.minute for event in day_events]
                person_key = f"{user_id}|{profile.role}"
                person_minutes[person_key].extend(minutes)
                person_days[person_key].add(current)
                role_minutes[profile.role].extend(minutes)
                role_days[profile.role].add((user_id, current))
                global_minutes.extend(minutes)
                global_days.add((user_id, current))
        current += timedelta(days=1)
    return prepared, pc_profiles


def _preprocessing_checksum(
    *,
    feature_config: Path,
    sequence_config: Path,
) -> str:
    payload = {
        "schema_version": "cert-preparation.v1",
        "feature_config_sha256": hashlib.sha256(feature_config.read_bytes()).hexdigest(),
        "sequence_config_sha256": hashlib.sha256(sequence_config.read_bytes()).hexdigest(),
        "timestamp_basis": "CERT_LOCAL_WALL_CLOCK",
        "event_order": ["timestamp", "source_rank", "event_uid"],
        "content_policy": "METADATA_ONLY",
        "window": "[D-29,D]",
        "last_day_scoring_only": True,
        "pc_map": "TRAIN_ONLY_FROZEN",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_scaler(path: Path, scaler: RobustFeatureScaler) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "robust-feature-scaler.v1",
                "location": scaler.location.tolist(),
                "scale": scaler.scale.tolist(),
                "support": scaler.support.tolist(),
                "checksum": scaler.checksum,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_scaler(path: Path) -> RobustFeatureScaler:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scaler = RobustFeatureScaler(
        location=np.asarray(payload["location"], dtype=np.float64),
        scale=np.asarray(payload["scale"], dtype=np.float64),
        support=np.asarray(payload["support"], dtype=np.int64),
        checksum=str(payload["checksum"]),
    )
    if scaler.location.shape != (FEATURE_DIMENSION,) or scaler.scale.shape != (
        FEATURE_DIMENSION,
    ):
        raise ValueError("scaler does not match Feature128")
    return scaler


def build_windows(
    *,
    prepared_days: dict[tuple[str, date], PreparedDay],
    directory: LdapDirectory,
    users: Sequence[str],
    start_day: date,
    end_day: date,
    split: str,
    feature_config: Path,
    sequence_config: Path,
    pc_profiles: dict[str, PCProfile],
    scaler: RobustFeatureScaler | None = None,
) -> tuple[PreparedWindows, RobustFeatureScaler]:
    split_upper = split.upper()
    samples = [
        (user_id, day)
        for day_offset in range((end_day - start_day).days + 1)
        for day in (start_day + timedelta(days=day_offset),)
        for user_id in users
        if locked_split_for_day(day) == split_upper and directory.profile(user_id, day)
    ]
    if not samples:
        raise ValueError(f"no eligible samples for split={split_upper}")

    shape = (len(samples), WINDOW_DAYS, FEATURE_DIMENSION)
    sequence_shape = (len(samples), WINDOW_DAYS, MAX_SEQUENCE_LENGTH)
    feature_values = np.zeros(shape, dtype=np.float32)
    feature_mask = np.zeros(shape, dtype=bool)
    tokens = np.zeros(sequence_shape, dtype=np.uint8)
    token_mask = np.zeros(sequence_shape, dtype=bool)
    pc_contexts = np.zeros(sequence_shape, dtype=np.uint8)
    calendar_contexts = np.zeros(sequence_shape, dtype=np.uint8)
    gap_buckets = np.zeros(sequence_shape, dtype=np.uint8)
    time_sin = np.zeros(sequence_shape, dtype=np.float32)
    time_cos = np.zeros(sequence_shape, dtype=np.float32)
    eligible_day_mask = np.zeros((len(samples), WINDOW_DAYS), dtype=bool)
    window_dates = np.empty((len(samples), WINDOW_DAYS), dtype="<U10")

    train_values: list[np.ndarray] = []
    train_masks: list[np.ndarray] = []
    for sample_index, (user_id, end) in enumerate(samples):
        for day_index in range(WINDOW_DAYS):
            current = end - timedelta(days=WINDOW_DAYS - 1 - day_index)
            window_dates[sample_index, day_index] = current.isoformat()
            row = prepared_days.get((user_id, current))
            if row is None:
                continue
            eligible_day_mask[sample_index, day_index] = True
            feature_values[sample_index, day_index] = row.values
            feature_mask[sample_index, day_index] = row.mask
            if locked_split_for_day(current) == "TRAIN":
                train_values.append(row.values)
                train_masks.append(row.mask)
            previous: datetime | None = None
            for event_index, event in enumerate(row.sequence):
                tokens[sample_index, day_index, event_index] = _event_token(event)
                token_mask[sample_index, day_index, event_index] = True
                context = classify_pc_context(
                    pc_profiles,
                    user_id=user_id,
                    pc=event.pc,
                )
                pc_contexts[sample_index, day_index, event_index] = PC_IDS[context]
                calendar_contexts[sample_index, day_index, event_index] = CALENDAR_IDS[
                    "WEEKEND" if current.weekday() >= 5 else "WEEKDAY"
                ]
                gap_minutes = (
                    0.0
                    if previous is None
                    else max(0.0, (event.timestamp - previous).total_seconds() / 60.0)
                )
                gap_buckets[sample_index, day_index, event_index] = _gap_bucket(gap_minutes)
                sin_value, cos_value = cyclic_time_components(event.minute)
                time_sin[sample_index, day_index, event_index] = sin_value
                time_cos[sample_index, day_index, event_index] = cos_value
                previous = event.timestamp

    if scaler is None:
        if not train_values:
            raise ValueError("a Train-fitted scaler is required when no Train rows are present")
        scaler = fit_robust_feature_scaler(
            np.stack(train_values),
            np.stack(train_masks),
        )
    feature_values = scaler.transform(feature_values, feature_mask)
    checksum = _preprocessing_checksum(
        feature_config=feature_config,
        sequence_config=sequence_config,
    )
    windows = PreparedWindows(
        feature_values=feature_values,
        feature_mask=feature_mask,
        tokens=tokens,
        token_mask=token_mask,
        pc_contexts=pc_contexts,
        calendar_contexts=calendar_contexts,
        gap_buckets=gap_buckets,
        time_sin=time_sin,
        time_cos=time_cos,
        window_dates=window_dates,
        eligible_day_mask=eligible_day_mask,
        sample_user_ids=np.asarray([item[0] for item in samples], dtype="<U32"),
        sample_end_days=np.asarray([item[1].isoformat() for item in samples], dtype="<U10"),
        sample_splits=np.asarray([split_upper] * len(samples), dtype="<U10"),
        feature_schema_version=np.asarray(FEATURE_SCHEMA_VERSION),
        sequence_schema_version=np.asarray(SEQUENCE_SCHEMA_VERSION),
        preprocessing_checksum=np.asarray(checksum),
        scaler_checksum=np.asarray(scaler.checksum),
        feature_value_space=np.asarray("robust_scaled"),
    )
    windows.validate()
    return windows, scaler


def save_windows(path: Path, windows: PreparedWindows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            field: getattr(windows, field)
            for field in windows.__dataclass_fields__
        },
    )
