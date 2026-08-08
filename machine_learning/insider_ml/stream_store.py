"""Disk-backed CERT preparation for machines with limited RAM.

Raw CSV files are consumed one source at a time and one calendar day at a
time.  Only metadata aggregates are persisted.  The large free-text ``content``
columns are not selected by Arrow and never enter Python objects or the store.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
import zlib
from collections import Counter, defaultdict, deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import fmean
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import numpy as np
import pyarrow.csv as arrow_csv

from insider_ml.artifacts import atomic_write_json
from insider_ml.cert_data import (
    CALENDAR_IDS,
    CERT_TIME_FORMAT,
    COUNT_FEATURES,
    PC_IDS,
    SOURCE_FILES,
    SOURCE_RANK,
    TOKEN_IDS,
    LdapDirectory,
    _gap_bucket,
    _ratio,
    _recipients,
    load_feature_names,
)
from insider_ml.contracts import (
    FEATURE_DIMENSION,
    FEATURE_SCHEMA_VERSION,
    MAX_SEQUENCE_LENGTH,
    SEQUENCE_SCHEMA_VERSION,
    TEST_END,
    TRAIN_END,
    TRAIN_START,
    WINDOW_DAYS,
    locked_split_for_day,
)
from insider_ml.preprocessing import (
    PCProfile,
    RobustFeatureScaler,
    classify_pc_context,
    fit_robust_feature_scaler,
    normalize_hostname,
)
from insider_ml.temporal import TemporalBaseline, unusual_time_relative_to_baseline

STORE_SCHEMA_VERSION = "cert-user-day-store.v1"
DAY_PAYLOAD_VERSION = 1
RAW_BLOCK_SIZE = 16 * 1024 * 1024
_DAY_HEADER = struct.Struct("<B")
_DAY_FEATURE_BYTES = FEATURE_DIMENSION * np.dtype("<f4").itemsize
_DAY_MASK_BYTES = math.ceil(FEATURE_DIMENSION / 8)
_DAY_SEQUENCE_BYTES = MAX_SEQUENCE_LENGTH
_DAY_SECONDS_BYTES = MAX_SEQUENCE_LENGTH * np.dtype("<u4").itemsize
_SOURCE_COLUMNS = {
    "LOGON": ("id", "date", "user", "pc", "activity"),
    "DEVICE": ("id", "date", "user", "pc", "activity"),
    "FILE": ("id", "date", "user", "pc", "filename"),
    "HTTP": ("id", "date", "user", "pc", "url"),
    "EMAIL": (
        "id",
        "date",
        "user",
        "pc",
        "to",
        "cc",
        "bcc",
        "from",
        "size",
        "attachments",
    ),
}
PRIMARY_DISABLED_FEATURES = {
    "source_transition_count",
    "distinct_source_transition_count",
    "source_transition_entropy",
    "inter_event_gap_p95_minutes",
}


def connect_store(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-131072")
    connection.execute("PRAGMA mmap_size=268435456")
    return connection


def initialize_store(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS source_days (
            source TEXT NOT NULL,
            user_id TEXT NOT NULL,
            day TEXT NOT NULL,
            event_count INTEGER NOT NULL,
            payload BLOB NOT NULL,
            PRIMARY KEY (source, user_id, day)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS ix_source_days_day
            ON source_days(day, user_id, source);
        CREATE TABLE IF NOT EXISTS pc_usage (
            source TEXT NOT NULL,
            pc TEXT NOT NULL,
            user_id TEXT NOT NULL,
            day TEXT NOT NULL,
            PRIMARY KEY (source, pc, user_id, day)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS ix_pc_usage_fit
            ON pc_usage(day, pc, user_id);
        CREATE TABLE IF NOT EXISTS pc_profiles (
            pc TEXT PRIMARY KEY,
            distinct_users INTEGER NOT NULL,
            dominance REAL NOT NULL,
            owner_user_id TEXT,
            is_shared INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS daily_tensors (
            user_id TEXT NOT NULL,
            day TEXT NOT NULL,
            split TEXT NOT NULL,
            role TEXT NOT NULL,
            role_epoch TEXT NOT NULL,
            event_count INTEGER NOT NULL,
            sequence_length INTEGER NOT NULL,
            feature_observed_count INTEGER NOT NULL,
            truncated INTEGER NOT NULL,
            payload BLOB NOT NULL,
            PRIMARY KEY (user_id, day)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS ix_daily_tensors_split_day
            ON daily_tensors(split, day, user_id);
        """
    )
    connection.execute(
        "INSERT OR IGNORE INTO metadata(key,value) VALUES('schema_version',?)",
        (STORE_SCHEMA_VERSION,),
    )
    connection.commit()


def metadata_get(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row[0])


def metadata_set(connection: sqlite3.Connection, key: str, value: object) -> None:
    rendered = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    connection.execute(
        """
        INSERT INTO metadata(key,value) VALUES(?,?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, rendered),
    )


def _stable_hash(value: str) -> int:
    digest = hashlib.blake2b(value.casefold().encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _basename_parts(filename: str) -> tuple[str, str, bool, int]:
    basename = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    head, separator, tail = basename.rpartition(".")
    has_extension = bool(separator and head and tail)
    extension = tail.casefold() if has_extension else ""
    stem = (head if has_extension else basename).casefold()
    return extension, stem, has_extension, len(basename)


def _candidate(
    *,
    second: int,
    event_uid: str,
    pc: str,
    token: int,
    ordinal: int,
) -> list[Any]:
    return [second, event_uid, pc, token, ordinal]


@dataclass(slots=True)
class SourceDayAccumulator:
    source: str
    user_id: str
    day: date
    count: int = 0
    seconds: list[int] = field(default_factory=list)
    pcs: Counter[str] = field(default_factory=Counter)
    head: list[list[Any]] = field(default_factory=list)
    tail: deque[list[Any]] = field(default_factory=lambda: deque(maxlen=128))
    data: dict[str, Any] = field(default_factory=dict)

    def _common(
        self,
        *,
        event_uid: str,
        timestamp: datetime,
        pc: str,
        token: int,
    ) -> int:
        second = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        ordinal = self.count
        self.count += 1
        self.seconds.append(second)
        if pc:
            self.pcs[pc] += 1
        item = _candidate(
            second=second,
            event_uid=event_uid,
            pc=pc,
            token=token,
            ordinal=ordinal,
        )
        if len(self.head) < 128:
            self.head.append(item)
        self.tail.append(item)
        return second

    def update(self, row: dict[str, Any], timestamp: datetime) -> None:
        event_uid = str(row.get("id") or "").strip()
        pc = str(row.get("pc") or "").strip()
        if self.source == "LOGON":
            action = str(row.get("activity") or "").strip().upper()
            token = TOKEN_IDS["LOGOFF" if action == "LOGOFF" else "LOGON"]
            second = self._common(
                event_uid=event_uid,
                timestamp=timestamp,
                pc=pc,
                token=token,
            )
            self.data.setdefault("events", []).append([second, action, pc])
            return
        if self.source == "DEVICE":
            action = str(row.get("activity") or "").strip().upper()
            token = TOKEN_IDS[
                "DEVICE_DISCONNECT" if action == "DISCONNECT" else "DEVICE_CONNECT"
            ]
            second = self._common(
                event_uid=event_uid,
                timestamp=timestamp,
                pc=pc,
                token=token,
            )
            self.data.setdefault("events", []).append([second, action, pc])
            return
        if self.source == "FILE":
            second = self._common(
                event_uid=event_uid,
                timestamp=timestamp,
                pc=pc,
                token=TOKEN_IDS["FILE"],
            )
            filename = str(row.get("filename") or "")
            extension, stem, has_extension, length = _basename_parts(filename)
            filename_hash = str(_stable_hash(filename))
            stem_hash = str(_stable_hash(stem))
            self.data.setdefault("filename_counts", Counter())[filename_hash] += 1
            self.data.setdefault("extension_counts", Counter())[extension] += 1
            self.data.setdefault("stem_hashes", set()).add(stem_hash)
            self.data["length_sum"] = self.data.get("length_sum", 0) + length
            self.data["length_sumsq"] = self.data.get("length_sumsq", 0) + length * length
            self.data["length_max"] = max(self.data.get("length_max", 0), length)
            if has_extension:
                ext_length = len(extension)
                self.data["extension_present"] = self.data.get("extension_present", 0) + 1
                self.data["extension_length_sum"] = (
                    self.data.get("extension_length_sum", 0) + ext_length
                )
                self.data["extension_length_max"] = max(
                    self.data.get("extension_length_max", 0),
                    ext_length,
                )
            else:
                self.data["extensionless"] = self.data.get("extensionless", 0) + 1
            basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
            self.data["multi_dot"] = self.data.get("multi_dot", 0) + int(
                basename.count(".") >= 2
            )
            previous = self.data.get("previous_second")
            if previous is not None:
                gap = max(0, second - previous)
                self.data["gap_count"] = self.data.get("gap_count", 0) + 1
                self.data["gap_sum_seconds"] = self.data.get("gap_sum_seconds", 0) + gap
                self.data["gap_min_seconds"] = min(
                    self.data.get("gap_min_seconds", gap),
                    gap,
                )
                in_burst = bool(self.data.get("in_burst", False))
                if gap <= 300 and not in_burst:
                    self.data["burst_count"] = self.data.get("burst_count", 0) + 1
                    self.data["in_burst"] = True
                elif gap > 300:
                    self.data["in_burst"] = False
            self.data["previous_second"] = second
            return
        if self.source == "HTTP":
            self._common(
                event_uid=event_uid,
                timestamp=timestamp,
                pc=pc,
                token=TOKEN_IDS["HTTP"],
            )
            url = str(row.get("url") or "")
            parsed = urlsplit(url if "://" in url else f"//{url}")
            host = normalize_hostname(url)
            url_hash = str(_stable_hash(url))
            path = parsed.path or "/"
            self.data.setdefault("url_counts", Counter())[url_hash] += 1
            self.data.setdefault("path_hashes", set()).add(str(_stable_hash(path)))
            if host:
                host_hash = str(_stable_hash(host))
                self.data.setdefault("host_counts", Counter())[host_hash] += 1
                self.data["host_length_sum"] = self.data.get("host_length_sum", 0) + len(
                    host
                )
                self.data["host_length_count"] = (
                    self.data.get("host_length_count", 0) + 1
                )
                self.data["host_length_max"] = max(
                    self.data.get("host_length_max", 0),
                    len(host),
                )
            scheme = parsed.scheme.lower()
            if scheme:
                self.data.setdefault("schemes", set()).add(scheme)
            self.data["https_count"] = self.data.get("https_count", 0) + int(
                scheme == "https"
            )
            self.data["query_present"] = self.data.get("query_present", 0) + int(
                bool(parsed.query)
            )
            query_keys = self.data.setdefault("query_keys", set())
            query_keys.update(
                str(_stable_hash(key.strip().lower()))
                for key, _value in parse_qsl(parsed.query, keep_blank_values=True)
                if key.strip()
            )
            self.data["url_length_sum"] = self.data.get("url_length_sum", 0) + len(url)
            self.data["url_length_max"] = max(
                self.data.get("url_length_max", 0),
                len(url),
            )
            self.data["path_depth_sum"] = self.data.get("path_depth_sum", 0) + sum(
                bool(segment) for segment in parsed.path.split("/")
            )
            return
        if self.source == "EMAIL":
            self._common(
                event_uid=event_uid,
                timestamp=timestamp,
                pc=pc,
                token=TOKEN_IDS["EMAIL"],
            )
            groups = {
                name: _recipients(str(row.get(name) or ""))
                for name in ("to", "cc", "bcc")
            }
            all_recipients = [
                recipient
                for values in groups.values()
                for recipient in values
            ]
            recipient_hashes = self.data.setdefault("recipient_hashes", set())
            external_hashes = self.data.setdefault("external_hashes", set())
            for recipient in all_recipients:
                recipient_hash = str(_stable_hash(recipient))
                recipient_hashes.add(recipient_hash)
                if recipient.endswith("@dtaa.com"):
                    self.data["internal_total"] = self.data.get("internal_total", 0) + 1
                else:
                    self.data["external_total"] = self.data.get("external_total", 0) + 1
                    external_hashes.add(recipient_hash)
            self.data["recipient_total"] = self.data.get("recipient_total", 0) + len(
                all_recipients
            )
            self.data["external_email_count"] = self.data.get(
                "external_email_count",
                0,
            ) + int(any(not value.endswith("@dtaa.com") for value in all_recipients))
            for name in ("to", "cc", "bcc"):
                key = f"{name}_total"
                self.data[key] = self.data.get(key, 0) + len(groups[name])
            self.data["bcc_email_count"] = self.data.get("bcc_email_count", 0) + int(
                bool(groups["bcc"])
            )
            attachment_count = max(0, int(row.get("attachments") or 0))
            self.data["attachment_email_count"] = self.data.get(
                "attachment_email_count",
                0,
            ) + int(attachment_count > 0)
            self.data["attachment_total"] = self.data.get("attachment_total", 0) + (
                attachment_count
            )
            self.data["attachment_max"] = max(
                self.data.get("attachment_max", 0),
                attachment_count,
            )
            size = max(0.0, float(row.get("size") or 0))
            self.data["size_sum"] = self.data.get("size_sum", 0.0) + size
            self.data["size_sumsq"] = self.data.get("size_sumsq", 0.0) + size * size
            self.data["size_max"] = max(self.data.get("size_max", 0.0), size)
            recipient_count = len(all_recipients)
            self.data["recipient_count_sum"] = self.data.get(
                "recipient_count_sum",
                0,
            ) + recipient_count
            self.data["recipient_count_sumsq"] = self.data.get(
                "recipient_count_sumsq",
                0,
            ) + recipient_count * recipient_count
            self.data["personal_from"] = self.data.get("personal_from", 0) + int(
                bool(str(row.get("from") or "").strip())
                and not str(row.get("from") or "").strip().lower().endswith("@dtaa.com")
            )
            return
        raise ValueError(f"unsupported source {self.source}")

    def payload(self) -> dict[str, Any]:
        candidates = self.head.copy()
        seen = {(item[4], item[0], item[1]) for item in candidates}
        candidates.extend(
            item
            for item in self.tail
            if (item[4], item[0], item[1]) not in seen
        )
        data = dict(self.data)
        for key, value in tuple(data.items()):
            if isinstance(value, Counter):
                data[key] = dict(value)
            elif isinstance(value, set):
                data[key] = sorted(value)
        data.pop("previous_second", None)
        data.pop("in_burst", None)
        return {
            "version": 1,
            "count": self.count,
            "seconds": self.seconds,
            "pcs": dict(self.pcs),
            "candidates": candidates,
            "data": data,
        }


def _compress_json(payload: dict[str, Any]) -> bytes:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return zlib.compress(encoded, level=6)


def _decompress_json(payload: bytes) -> dict[str, Any]:
    return json.loads(zlib.decompress(payload))


def _arrow_rows(path: Path, columns: Sequence[str]) -> Iterator[dict[str, Any]]:
    reader = arrow_csv.open_csv(
        path,
        read_options=arrow_csv.ReadOptions(block_size=RAW_BLOCK_SIZE, use_threads=True),
        convert_options=arrow_csv.ConvertOptions(include_columns=list(columns)),
    )
    for batch in reader:
        names = batch.schema.names
        values = [batch.column(index).to_pylist() for index in range(batch.num_columns)]
        for row_values in zip(*values, strict=True):
            yield dict(zip(names, row_values, strict=True))


def _raw_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def materialize_source(
    connection: sqlite3.Connection,
    *,
    raw_root: Path,
    source: str,
    max_rows: int | None = None,
) -> dict[str, object]:
    source_upper = source.upper()
    path = raw_root / SOURCE_FILES[source_upper]
    fingerprint = _raw_fingerprint(path)
    completion_key = f"source_complete_{source_upper}"
    prior = metadata_get(connection, completion_key)
    expected = json.dumps(
        {"fingerprint": fingerprint, "max_rows": max_rows},
        sort_keys=True,
    )
    if prior == expected:
        count = connection.execute(
            "SELECT COALESCE(SUM(event_count),0) FROM source_days WHERE source=?",
            (source_upper,),
        ).fetchone()[0]
        return {"source": source_upper, "status": "SKIPPED", "events": int(count)}

    connection.execute("DELETE FROM source_days WHERE source=?", (source_upper,))
    connection.execute("DELETE FROM pc_usage WHERE source=?", (source_upper,))
    connection.commit()
    current_day: date | None = None
    accumulators: dict[str, SourceDayAccumulator] = {}
    rows_read = 0
    rows_written = 0

    def flush() -> None:
        nonlocal rows_written
        if not accumulators:
            return
        source_rows = []
        usage_rows = []
        for user_id, accumulator in accumulators.items():
            payload = accumulator.payload()
            source_rows.append(
                (
                    source_upper,
                    user_id,
                    accumulator.day.isoformat(),
                    accumulator.count,
                    _compress_json(payload),
                )
            )
            usage_rows.extend(
                (
                    source_upper,
                    pc,
                    user_id,
                    accumulator.day.isoformat(),
                )
                for pc in accumulator.pcs
            )
        connection.executemany(
            """
            INSERT INTO source_days(source,user_id,day,event_count,payload)
            VALUES(?,?,?,?,?)
            """,
            source_rows,
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO pc_usage(source,pc,user_id,day)
            VALUES(?,?,?,?)
            """,
            usage_rows,
        )
        connection.commit()
        rows_written += len(source_rows)

    for row in _arrow_rows(path, _SOURCE_COLUMNS[source_upper]):
        if max_rows is not None and rows_read >= max_rows:
            break
        rows_read += 1
        timestamp = datetime.strptime(str(row["date"]), CERT_TIME_FORMAT)
        if current_day is None:
            current_day = timestamp.date()
        elif timestamp.date() != current_day:
            if timestamp.date() < current_day:
                raise ValueError(f"{path} is not ordered by local timestamp")
            flush()
            accumulators.clear()
            current_day = timestamp.date()
        user_id = str(row.get("user") or "").strip()
        if not user_id:
            continue
        accumulator = accumulators.get(user_id)
        if accumulator is None:
            accumulator = SourceDayAccumulator(
                source=source_upper,
                user_id=user_id,
                day=timestamp.date(),
            )
            accumulators[user_id] = accumulator
        accumulator.update(row, timestamp)
    flush()
    metadata_set(connection, completion_key, expected)
    connection.commit()
    return {
        "source": source_upper,
        "status": "BUILT",
        "raw_rows": rows_read,
        "user_days": rows_written,
        "content_selected": False,
    }


def fit_store_pc_profiles(connection: sqlite3.Connection) -> dict[str, int]:
    connection.execute("DELETE FROM pc_profiles")
    query = connection.execute(
        """
        SELECT pc,user_id,COUNT(DISTINCT day) AS user_days
        FROM pc_usage
        WHERE day BETWEEN ? AND ?
        GROUP BY pc,user_id
        ORDER BY pc,user_id
        """,
        (TRAIN_START.isoformat(), TRAIN_END.isoformat()),
    )
    current_pc: str | None = None
    counts: dict[str, int] = {}
    inserted = 0

    def flush() -> None:
        nonlocal inserted
        if current_pc is None or not counts:
            return
        total = sum(counts.values())
        highest = max(counts.values())
        dominance = highest / total
        candidates = sorted(user for user, count in counts.items() if count == highest)
        is_shared = len(counts) >= 5 and dominance < 0.5
        owner = candidates[0] if not is_shared and dominance >= 0.5 else None
        connection.execute(
            """
            INSERT INTO pc_profiles(
                pc,distinct_users,dominance,owner_user_id,is_shared
            ) VALUES(?,?,?,?,?)
            """,
            (current_pc, len(counts), dominance, owner, int(is_shared)),
        )
        inserted += 1

    for pc, user_id, user_days in query:
        pc = str(pc)
        if current_pc is not None and pc != current_pc:
            flush()
            counts = {}
        current_pc = pc
        counts[str(user_id)] = int(user_days)
    flush()
    connection.commit()
    metadata_set(connection, "pc_profiles_fit_split", "TRAIN")
    metadata_set(connection, "pc_profiles_frozen", "true")
    connection.commit()
    return {"pc_profiles": inserted}


def load_store_pc_profiles(connection: sqlite3.Connection) -> dict[str, PCProfile]:
    return {
        str(row[0]): PCProfile(
            pc=str(row[0]),
            distinct_users=int(row[1]),
            dominance=float(row[2]),
            owner_user_id=str(row[3]) if row[3] is not None else None,
            is_shared=bool(row[4]),
        )
        for row in connection.execute(
            """
            SELECT pc,distinct_users,dominance,owner_user_id,is_shared
            FROM pc_profiles
            """
        )
    }


@dataclass(slots=True)
class RollingTemporalReference:
    histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(1440, dtype=np.int64)
    )
    entries: deque[tuple[date, str, np.ndarray]] = field(default_factory=deque)
    users: Counter[str] = field(default_factory=Counter)

    def expire(self, current_day: date) -> None:
        while self.entries and (current_day - self.entries[0][0]).days > 30:
            _day, user_id, histogram = self.entries.popleft()
            self.histogram -= histogram
            self.users[user_id] -= 1
            if self.users[user_id] <= 0:
                del self.users[user_id]

    def add(
        self,
        *,
        day: date,
        user_id: str,
        histogram: np.ndarray,
    ) -> None:
        self.entries.append((day, user_id, histogram))
        self.histogram += histogram
        self.users[user_id] += 1

    @property
    def user_days(self) -> int:
        return len(self.entries)

    def frozen_copy(self) -> RollingTemporalReference:
        result = RollingTemporalReference()
        result.histogram = self.histogram.copy()
        result.users = self.users.copy()
        return result


class TemporalReferenceState:
    def __init__(self) -> None:
        self.person: dict[str, RollingTemporalReference] = {}
        self.role: dict[str, RollingTemporalReference] = {}
        self.global_ = RollingTemporalReference()
        self.frozen_person: dict[str, RollingTemporalReference] | None = None
        self.frozen_role: dict[str, RollingTemporalReference] | None = None
        self.frozen_global: RollingTemporalReference | None = None
        self._baseline_cache: dict[tuple[str, str, str], TemporalBaseline] = {}

    @staticmethod
    def _person_key(user_id: str, role_epoch: str) -> str:
        return f"{user_id}|{role_epoch}"

    def expire(self, current_day: date) -> None:
        for reference in self.person.values():
            reference.expire(current_day)
        for reference in self.role.values():
            reference.expire(current_day)
        self.global_.expire(current_day)
        self._baseline_cache.clear()

    def freeze(self) -> None:
        self.frozen_person = {
            key: value.frozen_copy() for key, value in self.person.items()
        }
        self.frozen_role = {
            key: value.frozen_copy() for key, value in self.role.items()
        }
        self.frozen_global = self.global_.frozen_copy()
        self._baseline_cache.clear()

    def add_train_day(
        self,
        *,
        day: date,
        user_id: str,
        role: str,
        role_epoch: str,
        seconds: Sequence[int],
    ) -> None:
        if not seconds:
            return
        histogram = np.bincount(
            np.asarray(seconds, dtype=np.int64) // 60,
            minlength=1440,
        ).astype(np.int64)
        person_key = self._person_key(user_id, role_epoch)
        self.person.setdefault(person_key, RollingTemporalReference()).add(
            day=day,
            user_id=user_id,
            histogram=histogram,
        )
        self.role.setdefault(role, RollingTemporalReference()).add(
            day=day,
            user_id=user_id,
            histogram=histogram,
        )
        self.global_.add(day=day, user_id=user_id, histogram=histogram)

    @staticmethod
    def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        order = np.argsort(values)
        ordered_values = values[order]
        ordered_weights = weights[order]
        cutoff = ordered_weights.sum() / 2.0
        index = int(np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left"))
        return float(ordered_values[min(index, len(ordered_values) - 1)])

    @classmethod
    def _fit(
        cls,
        reference: RollingTemporalReference,
        *,
        level: str,
        scope_key: str,
        fitted_through: date,
    ) -> TemporalBaseline | None:
        nonzero = np.flatnonzero(reference.histogram)
        if nonzero.size == 0:
            return None
        weights = reference.histogram[nonzero].astype(np.float64)
        minutes = nonzero.astype(np.float64) + 0.5
        angles = 2.0 * np.pi * minutes / 1440.0
        mean_sin = float(np.average(np.sin(angles), weights=weights))
        mean_cos = float(np.average(np.cos(angles), weights=weights))
        if math.hypot(mean_sin, mean_cos) < 1e-8:
            return None
        center_angle = math.atan2(mean_sin, mean_cos) % (2.0 * np.pi)
        center = center_angle * 1440.0 / (2.0 * np.pi)
        raw_distance = np.abs(minutes - center)
        distances = np.minimum(raw_distance, 1440.0 - raw_distance)
        location = cls._weighted_median(distances, weights)
        absolute_deviation = np.abs(distances - location)
        mad = cls._weighted_median(absolute_deviation, weights)
        return TemporalBaseline(
            level=level,
            scope_key=scope_key,
            fitted_through=fitted_through,
            center_minute=center,
            distance_location=location,
            distance_scale=max(1.4826 * mad, 1.0),
            observation_count=int(weights.sum()),
        )

    def select(
        self,
        *,
        day: date,
        user_id: str,
        role: str,
        role_epoch: str,
    ) -> TemporalBaseline | None:
        frozen = day > TRAIN_END
        person_map = self.frozen_person if frozen else self.person
        role_map = self.frozen_role if frozen else self.role
        global_reference = self.frozen_global if frozen else self.global_
        if person_map is None or role_map is None or global_reference is None:
            return None
        person_key = self._person_key(user_id, role_epoch)
        candidates: list[tuple[str, str, RollingTemporalReference]] = []
        person = person_map.get(person_key)
        if person is not None and person.user_days >= 30:
            candidates.append(("PERSON", person_key, person))
        role_reference = role_map.get(role)
        if (
            role_reference is not None
            and len(
                {
                    peer
                    for peer in role_reference.users
                    if peer != user_id
                }
            )
            >= 15
            and role_reference.user_days >= 300
        ):
            candidates.append(("ROLE", role, role_reference))
        if len(global_reference.users) >= 200 and global_reference.user_days >= 10000:
            candidates.append(("GLOBAL", "GLOBAL", global_reference))
        fitted_through = TRAIN_END if frozen else day - timedelta(days=1)
        for level, scope_key, reference in candidates:
            cache_key = (level, scope_key, fitted_through.isoformat())
            baseline = self._baseline_cache.get(cache_key)
            if baseline is None:
                baseline = self._fit(
                    reference,
                    level=level,
                    scope_key=scope_key,
                    fitted_through=fitted_through,
                )
                if baseline is not None:
                    self._baseline_cache[cache_key] = baseline
            if baseline is not None:
                return baseline
        return None


@dataclass(slots=True)
class UserFeatureHistory:
    event_days: deque[tuple[date, int]] = field(default_factory=deque)
    domain_days: deque[tuple[date, set[str], int]] = field(default_factory=deque)
    recipient_days: deque[tuple[date, set[str]]] = field(default_factory=deque)
    last_active_day: date | None = None

    def expire(self, current_day: date) -> None:
        while self.event_days and (current_day - self.event_days[0][0]).days > 30:
            self.event_days.popleft()
        while self.domain_days and (current_day - self.domain_days[0][0]).days > 30:
            self.domain_days.popleft()
        while (
            self.recipient_days
            and (current_day - self.recipient_days[0][0]).days > 30
        ):
            self.recipient_days.popleft()


def _payload_count(payload: dict[str, Any] | None) -> int:
    return 0 if payload is None else int(payload["count"])


def _payload_seconds(payload: dict[str, Any] | None) -> list[int]:
    return [] if payload is None else [int(value) for value in payload["seconds"]]


def _source_pc_counts(
    payload: dict[str, Any] | None,
    *,
    pc_profiles: dict[str, PCProfile],
    user_id: str,
) -> Counter[str]:
    result: Counter[str] = Counter()
    if payload is None:
        return result
    for pc, count in payload["pcs"].items():
        context = classify_pc_context(pc_profiles, user_id=user_id, pc=pc)
        result[context] += int(count)
    return result


def _action_pc_counts(
    payload: dict[str, Any] | None,
    *,
    action: str,
    pc_profiles: dict[str, PCProfile],
    user_id: str,
) -> Counter[str]:
    result: Counter[str] = Counter()
    if payload is None:
        return result
    for _second, observed_action, pc in payload["data"].get("events", []):
        if str(observed_action) != action:
            continue
        context = classify_pc_context(pc_profiles, user_id=user_id, pc=str(pc))
        result[context] += 1
    return result


def _paired_seconds(
    payload: dict[str, Any] | None,
    *,
    start_action: str,
    end_action: str,
) -> tuple[list[float], int, int]:
    if payload is None:
        return [], 0, 0
    open_by_pc: dict[str, deque[int]] = defaultdict(deque)
    durations: list[float] = []
    unmatched_end = 0
    for second, action, pc in payload["data"].get("events", []):
        if action == start_action:
            open_by_pc[str(pc)].append(int(second))
        elif action == end_action:
            if not open_by_pc[str(pc)]:
                unmatched_end += 1
            else:
                durations.append(
                    max(0.0, (int(second) - open_by_pc[str(pc)].popleft()) / 60.0)
                )
    return durations, sum(len(value) for value in open_by_pc.values()), unmatched_end


def _counter_entropy(raw: dict[str, Any]) -> float:
    counts = [int(value) for value in raw.values()]
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts)


def _population_std(*, total: float, total_squares: float, count: int) -> float | None:
    if count == 0:
        return None
    variance = max(0.0, total_squares / count - (total / count) ** 2)
    return math.sqrt(variance)


def _timeline(payloads: dict[str, dict[str, Any]]) -> list[tuple[int, int, str, int]]:
    events = [
        (int(second), SOURCE_RANK[source], source, ordinal)
        for source, payload in payloads.items()
        for ordinal, second in enumerate(payload["seconds"])
    ]
    events.sort()
    return events


def _cross_features(
    timeline: Sequence[tuple[int, int, str, int]],
) -> dict[str, float | None]:
    if len(timeline) < 2:
        return {
            "source_transition_count": 0.0,
            "distinct_source_transition_count": 0.0,
            "source_transition_entropy": 0.0,
            "inter_event_gap_p95_minutes": None,
            "multi_channel_burst_count": 0.0,
        }
    pairs = [
        (left[2], right[2])
        for left, right in zip(timeline, timeline[1:], strict=False)
        if left[2] != right[2]
    ]
    counts = Counter(pairs)
    total = sum(counts.values())
    gaps = sorted(
        max(0, right[0] - left[0]) / 60.0
        for left, right in zip(timeline, timeline[1:], strict=False)
    )
    p95_index = max(0, math.ceil(0.95 * len(gaps)) - 1)
    burst_count = 0
    run_sources = {timeline[0][2]}
    in_run = False
    for left, right in zip(timeline, timeline[1:], strict=False):
        if right[0] - left[0] <= 300:
            run_sources.add(right[2])
            in_run = True
        else:
            if in_run and len(run_sources) >= 2:
                burst_count += 1
            run_sources = {right[2]}
            in_run = False
    if in_run and len(run_sources) >= 2:
        burst_count += 1
    return {
        "source_transition_count": float(total),
        "distinct_source_transition_count": float(len(counts)),
        "source_transition_entropy": (
            -sum((count / total) * math.log2(count / total) for count in counts.values())
            if total
            else 0.0
        ),
        "inter_event_gap_p95_minutes": gaps[p95_index],
        "multi_channel_burst_count": float(burst_count),
    }


def _features_from_payloads(
    *,
    user_id: str,
    day: date,
    role_start: date,
    payloads: dict[str, dict[str, Any]],
    feature_names: Sequence[str],
    pc_profiles: dict[str, PCProfile],
    history: UserFeatureHistory,
    temporal_baseline: TemporalBaseline | None,
) -> tuple[np.ndarray, np.ndarray]:
    values: dict[str, float | None] = {name: None for name in feature_names}
    for name in COUNT_FEATURES:
        if name in values:
            values[name] = 0.0
    weekend = day.weekday() >= 5
    source_counts = {
        source: _payload_count(payloads.get(source)) for source in SOURCE_FILES
    }
    all_seconds = [
        second
        for source in SOURCE_FILES
        for second in _payload_seconds(payloads.get(source))
    ]

    logon_payload = payloads.get("LOGON")
    logon_events = (
        [] if logon_payload is None else logon_payload["data"].get("events", [])
    )
    logon_seconds = [
        int(second) for second, action, _pc in logon_events if action == "LOGON"
    ]
    logoff_seconds = [
        int(second) for second, action, _pc in logon_events if action == "LOGOFF"
    ]
    logon_context = _action_pc_counts(
        logon_payload,
        action="LOGON",
        pc_profiles=pc_profiles,
        user_id=user_id,
    )
    logon_durations, _unmatched_logon, _unmatched_logoff = _paired_seconds(
        logon_payload,
        start_action="LOGON",
        end_action="LOGOFF",
    )
    first_logon = min(logon_seconds, default=None)
    last_logon = max(logon_seconds, default=None)
    last_logoff = max(logoff_seconds, default=None)
    work_end = last_logoff if last_logoff is not None else last_logon
    values.update(
        {
            "logon_count": float(len(logon_seconds)),
            "logoff_count": float(len(logoff_seconds)),
            "weekend_logon_count": float(len(logon_seconds) if weekend else 0),
            "distinct_logon_pc_count": float(
                len(
                    {
                        str(pc)
                        for _second, action, pc in logon_events
                        if action == "LOGON"
                    }
                )
            ),
            "own_pc_logon_count": float(logon_context["OWN"]),
            "shared_pc_logon_count": float(logon_context["SHARED"]),
            "foreign_pc_logon_count": float(logon_context["FOREIGN"]),
            "foreign_pc_logon_ratio": _ratio(
                logon_context["FOREIGN"],
                len(logon_seconds),
            ),
            "first_logon_minute": (
                first_logon / 60.0 if first_logon is not None else None
            ),
            "last_logon_minute": (
                last_logon / 60.0 if last_logon is not None else None
            ),
            "last_logoff_minute": (
                last_logoff / 60.0 if last_logoff is not None else None
            ),
            "work_span_minutes": (
                max(0.0, work_end - first_logon) / 60.0
                if first_logon is not None and work_end is not None
                else None
            ),
            "matched_logon_session_count": float(len(logon_durations)),
            "total_logon_session_minutes": (
                sum(logon_durations) if logon_durations else None
            ),
            "max_logon_session_minutes": (
                max(logon_durations) if logon_durations else None
            ),
        }
    )

    device_payload = payloads.get("DEVICE")
    device_events = (
        [] if device_payload is None else device_payload["data"].get("events", [])
    )
    connect_count = sum(action == "CONNECT" for _second, action, _pc in device_events)
    disconnect_count = sum(
        action == "DISCONNECT" for _second, action, _pc in device_events
    )
    device_context = _action_pc_counts(
        device_payload,
        action="CONNECT",
        pc_profiles=pc_profiles,
        user_id=user_id,
    )
    device_durations, unmatched_connect, unmatched_disconnect = _paired_seconds(
        device_payload,
        start_action="CONNECT",
        end_action="DISCONNECT",
    )
    values.update(
        {
            "device_connect_count": float(connect_count),
            "device_disconnect_count": float(disconnect_count),
            "device_session_count": float(len(device_durations)),
            "unmatched_connect_count": float(unmatched_connect),
            "unmatched_disconnect_count": float(unmatched_disconnect),
            "weekend_device_connect_count": float(connect_count if weekend else 0),
            "own_pc_device_connect_count": float(device_context["OWN"]),
            "shared_pc_device_connect_count": float(device_context["SHARED"]),
            "foreign_pc_device_connect_count": float(device_context["FOREIGN"]),
            "device_active_minutes": (
                sum(device_durations) if device_durations else None
            ),
            "max_device_session_minutes": (
                max(device_durations) if device_durations else None
            ),
        }
    )

    file_payload = payloads.get("FILE")
    file_count = source_counts["FILE"]
    file_data = {} if file_payload is None else file_payload["data"]
    filename_counts = file_data.get("filename_counts", {})
    extension_counts = file_data.get("extension_counts", {})
    nonempty_extensions = {
        key: int(value) for key, value in extension_counts.items() if key
    }
    filename_length_sum = float(file_data.get("length_sum", 0))
    filename_length_sumsq = float(file_data.get("length_sumsq", 0))
    extension_present = int(file_data.get("extension_present", 0))
    file_context = _source_pc_counts(
        file_payload,
        pc_profiles=pc_profiles,
        user_id=user_id,
    )
    values.update(
        {
            "file_copy_count": float(file_count),
            "distinct_filename_count": float(len(filename_counts)),
            "distinct_extension_count": float(len(nonempty_extensions)),
            "filename_length_max": (
                float(file_data.get("length_max", 0)) if file_count else None
            ),
            "filename_length_std": _population_std(
                total=filename_length_sum,
                total_squares=filename_length_sumsq,
                count=file_count,
            ),
            "extension_length_mean": (
                float(file_data.get("extension_length_sum", 0)) / extension_present
                if extension_present
                else None
            ),
            "extension_length_max": (
                float(file_data.get("extension_length_max", 0))
                if extension_present
                else None
            ),
            "extensionless_file_count": float(file_data.get("extensionless", 0)),
            "multi_dot_filename_count": float(file_data.get("multi_dot", 0)),
            "distinct_filename_stem_count": float(
                len(file_data.get("stem_hashes", []))
            ),
            "repeated_extension_copy_count": float(
                sum(max(count - 1, 0) for count in nonempty_extensions.values())
            ),
            "extension_present_ratio": _ratio(extension_present, file_count),
            "weekend_file_count": float(file_count if weekend else 0),
            "own_pc_file_count": float(file_context["OWN"]),
            "shared_pc_file_count": float(file_context["SHARED"]),
            "foreign_pc_file_count": float(file_context["FOREIGN"]),
            "distinct_file_pc_count": float(
                len(file_payload["pcs"]) if file_payload is not None else 0
            ),
            "filename_length_mean": (
                filename_length_sum / file_count if file_count else None
            ),
            "file_copy_interarrival_mean_minutes": (
                float(file_data.get("gap_sum_seconds", 0))
                / int(file_data.get("gap_count", 0))
                / 60.0
                if int(file_data.get("gap_count", 0))
                else None
            ),
            "file_copy_interarrival_min_minutes": (
                float(file_data["gap_min_seconds"]) / 60.0
                if "gap_min_seconds" in file_data
                else None
            ),
            "repeated_filename_copy_count": float(
                sum(max(int(count) - 1, 0) for count in filename_counts.values())
            ),
            "file_extension_entropy": _counter_entropy(nonempty_extensions),
            "file_copy_burst_count_5m": float(file_data.get("burst_count", 0)),
        }
    )

    http_payload = payloads.get("HTTP")
    http_count = source_counts["HTTP"]
    http_data = {} if http_payload is None else http_payload["data"]
    host_counts = http_data.get("host_counts", {})
    url_counts = http_data.get("url_counts", {})
    prior_hosts = {
        host
        for _history_day, hosts, _event_count in history.domain_days
        for host in hosts
    }
    prior_http_count = sum(item[2] for item in history.domain_days)
    http_context = _source_pc_counts(
        http_payload,
        pc_profiles=pc_profiles,
        user_id=user_id,
    )
    values.update(
        {
            "http_request_count": float(http_count),
            "distinct_url_count": float(len(url_counts)),
            "distinct_domain_count": float(len(host_counts)),
            "distinct_path_count": float(len(http_data.get("path_hashes", []))),
            "weekend_http_count": float(http_count if weekend else 0),
            "own_pc_http_count": float(http_context["OWN"]),
            "shared_pc_http_count": float(http_context["SHARED"]),
            "foreign_pc_http_count": float(http_context["FOREIGN"]),
            "distinct_hostname_count": float(len(host_counts)),
            "distinct_scheme_count": float(len(http_data.get("schemes", []))),
            "https_request_count": float(http_data.get("https_count", 0)),
            "query_present_request_count": float(
                http_data.get("query_present", 0)
            ),
            "distinct_query_key_count": float(
                len(http_data.get("query_keys", []))
            ),
            "repeated_hostname_request_count": float(
                sum(max(int(count) - 1, 0) for count in host_counts.values())
            ),
            "hostname_length_mean": (
                float(http_data.get("host_length_sum", 0))
                / int(http_data.get("host_length_count", 0))
                if int(http_data.get("host_length_count", 0))
                else None
            ),
            "hostname_length_max": (
                float(http_data.get("host_length_max", 0)) if host_counts else None
            ),
            "url_length_mean": (
                float(http_data.get("url_length_sum", 0)) / http_count
                if http_count
                else None
            ),
            "url_length_max": (
                float(http_data.get("url_length_max", 0)) if http_count else None
            ),
            "url_path_depth_mean": (
                float(http_data.get("path_depth_sum", 0)) / http_count
                if http_count
                else None
            ),
            "distinct_url_path_count": float(
                len(http_data.get("path_hashes", []))
            ),
            "domain_entropy": _counter_entropy(host_counts),
            "repeated_url_request_count": float(
                sum(max(int(count) - 1, 0) for count in url_counts.values())
            ),
            "new_domain_count_30d": (
                float(len(set(host_counts) - prior_hosts))
                if prior_http_count >= 20
                else None
            ),
        }
    )

    email_payload = payloads.get("EMAIL")
    email_count = source_counts["EMAIL"]
    email_data = {} if email_payload is None else email_payload["data"]
    email_context = _source_pc_counts(
        email_payload,
        pc_profiles=pc_profiles,
        user_id=user_id,
    )
    recipient_total = int(email_data.get("recipient_total", 0))
    external_total = int(email_data.get("external_total", 0))
    prior_external = {
        recipient
        for _history_day, recipients in history.recipient_days
        for recipient in recipients
    }
    prior_external_observations = sum(
        len(recipients) for _history_day, recipients in history.recipient_days
    )
    size_sum = float(email_data.get("size_sum", 0))
    size_sumsq = float(email_data.get("size_sumsq", 0))
    recipient_count_sum = float(email_data.get("recipient_count_sum", 0))
    recipient_count_sumsq = float(email_data.get("recipient_count_sumsq", 0))
    values.update(
        {
            "email_sent_count": float(email_count),
            "weekend_email_count": float(email_count if weekend else 0),
            "own_pc_email_count": float(email_context["OWN"]),
            "shared_pc_email_count": float(email_context["SHARED"]),
            "foreign_pc_email_count": float(email_context["FOREIGN"]),
            "recipient_total_count": float(recipient_total),
            "unique_recipient_count": float(
                len(email_data.get("recipient_hashes", []))
            ),
            "internal_recipient_total": float(email_data.get("internal_total", 0)),
            "external_recipient_total": float(external_total),
            "external_recipient_email_count": float(
                email_data.get("external_email_count", 0)
            ),
            "external_recipient_ratio": _ratio(external_total, recipient_total),
            "to_recipient_total": float(email_data.get("to_total", 0)),
            "cc_recipient_total": float(email_data.get("cc_total", 0)),
            "bcc_recipient_total": float(email_data.get("bcc_total", 0)),
            "bcc_email_count": float(email_data.get("bcc_email_count", 0)),
            "attachment_email_count": float(
                email_data.get("attachment_email_count", 0)
            ),
            "attachment_total_count": float(
                email_data.get("attachment_total", 0)
            ),
            "attachment_max_count": float(email_data.get("attachment_max", 0)),
            "email_size_total": size_sum if email_count else None,
            "email_size_mean": size_sum / email_count if email_count else None,
            "email_size_max": (
                float(email_data.get("size_max", 0)) if email_count else None
            ),
            "personal_from_email_count": float(
                email_data.get("personal_from", 0)
            ),
            "recipient_count_mean": (
                recipient_count_sum / email_count if email_count else None
            ),
            "email_size_std": _population_std(
                total=size_sum,
                total_squares=size_sumsq,
                count=email_count,
            ),
            "new_external_recipient_count_30d": (
                float(
                    len(
                        set(email_data.get("external_hashes", []))
                        - prior_external
                    )
                )
                if prior_external_observations >= 5
                else None
            ),
        }
    )
    # Retain this computed value for formula auditing even though Feature128
    # currently exposes only recipient_count_mean, not its standard deviation.
    _ = _population_std(
        total=recipient_count_sum,
        total_squares=recipient_count_sumsq,
        count=email_count,
    )

    event_count = sum(source_counts.values())
    all_context: Counter[str] = Counter()
    all_pcs: set[str] = set()
    for payload in payloads.values():
        all_context.update(
            _source_pc_counts(
                payload,
                pc_profiles=pc_profiles,
                user_id=user_id,
            )
        )
        all_pcs.update(payload["pcs"])
    timeline = _timeline(payloads)
    values.update(_cross_features(timeline))
    values.update(
        {
            "total_event_count": float(event_count),
            "active_channel_count": float(sum(count > 0 for count in source_counts.values())),
            "weekend_event_count": float(event_count if weekend else 0),
            "weekend_event_ratio": _ratio(event_count if weekend else 0, event_count),
            "own_pc_event_count": float(all_context["OWN"]),
            "shared_pc_event_count": float(all_context["SHARED"]),
            "foreign_pc_event_count": float(all_context["FOREIGN"]),
            "foreign_pc_event_ratio": _ratio(all_context["FOREIGN"], event_count),
            "distinct_all_pc_count": float(len(all_pcs)),
        }
    )

    recent_7 = [
        count
        for history_day, count in history.event_days
        if 0 < (day - history_day).days <= 7
    ]
    recent_30 = [
        count
        for history_day, count in history.event_days
        if 0 < (day - history_day).days <= 30
    ]
    values.update(
        {
            "active_days_last_7": float(sum(count > 0 for count in recent_7)),
            "active_days_last_30": float(sum(count > 0 for count in recent_30)),
            "total_events_mean_7d": fmean(recent_7) if recent_7 else 0.0,
            "total_events_mean_30d": fmean(recent_30) if recent_30 else 0.0,
            "total_events_std_30d": (
                float(np.std(recent_30, ddof=0)) if recent_30 else 0.0
            ),
            "days_since_last_active_day": (
                float((day - history.last_active_day).days)
                if history.last_active_day is not None
                else None
            ),
            "days_in_current_role": float((day - role_start).days + 1),
        }
    )
    values["total_events_short_long_ratio"] = _ratio(
        float(values["total_events_mean_7d"] or 0),
        float(values["total_events_mean_30d"] or 0),
    )

    if temporal_baseline is not None and all_seconds:
        deviations = [
            unusual_time_relative_to_baseline(second / 60.0, temporal_baseline)
            for second in all_seconds
        ]
        values["unusual_time_relative_to_baseline_mean"] = fmean(deviations)
        values["unusual_time_relative_to_baseline_max"] = max(deviations)
        for source, feature_name in (
            ("DEVICE", "device_time_deviation_mean"),
            ("FILE", "file_time_deviation_mean"),
            ("HTTP", "http_time_deviation_mean"),
            ("EMAIL", "email_time_deviation_mean"),
        ):
            source_seconds = _payload_seconds(payloads.get(source))
            values[feature_name] = (
                fmean(
                    unusual_time_relative_to_baseline(
                        second / 60.0,
                        temporal_baseline,
                    )
                    for second in source_seconds
                )
                if source_seconds
                else None
            )
        if first_logon is not None:
            values["first_logon_time_deviation"] = unusual_time_relative_to_baseline(
                first_logon / 60.0,
                temporal_baseline,
            )
        if last_logoff is not None:
            values["last_logoff_time_deviation"] = unusual_time_relative_to_baseline(
                last_logoff / 60.0,
                temporal_baseline,
            )
        session_seconds = [*logon_seconds, *logoff_seconds]
        if session_seconds:
            values["session_time_deviation_max"] = max(
                unusual_time_relative_to_baseline(
                    second / 60.0,
                    temporal_baseline,
                )
                for second in session_seconds
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
        if math.isfinite(numeric):
            vector[index] = numeric
            mask[index] = True
    return vector, mask


def _sequence_from_payloads(
    *,
    user_id: str,
    day: date,
    payloads: dict[str, dict[str, Any]],
    pc_profiles: dict[str, PCProfile],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    bool,
]:
    all_candidates: list[tuple[int, int, str, str, int, int]] = []
    for source, payload in payloads.items():
        for second, event_uid, pc, token, ordinal in payload["candidates"]:
            all_candidates.append(
                (
                    int(second),
                    SOURCE_RANK[source],
                    str(event_uid),
                    str(pc),
                    int(token),
                    int(ordinal),
                )
            )
    all_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    total_events = sum(_payload_count(payload) for payload in payloads.values())
    if len(all_candidates) > MAX_SEQUENCE_LENGTH:
        selected = [*all_candidates[:128], *all_candidates[-128:]]
    else:
        selected = all_candidates

    timeline = _timeline(payloads)
    previous_by_source_ordinal: dict[tuple[int, int], int | None] = {}
    previous_second: int | None = None
    for second, rank, _source, ordinal in timeline:
        previous_by_source_ordinal[(rank, ordinal)] = previous_second
        previous_second = second

    tokens = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.uint8)
    pcs = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.uint8)
    calendars = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.uint8)
    gaps = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.uint8)
    seconds = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.uint32)
    for index, (second, rank, _event_uid, pc, token, ordinal) in enumerate(selected):
        tokens[index] = token
        pcs[index] = PC_IDS[
            classify_pc_context(pc_profiles, user_id=user_id, pc=pc)
        ]
        calendars[index] = CALENDAR_IDS[
            "WEEKEND" if day.weekday() >= 5 else "WEEKDAY"
        ]
        prior = previous_by_source_ordinal[(rank, ordinal)]
        elapsed = 0.0 if prior is None else max(0, second - prior) / 60.0
        gaps[index] = _gap_bucket(elapsed)
        seconds[index] = second
    return (
        tokens,
        pcs,
        calendars,
        gaps,
        seconds,
        len(selected),
        total_events > MAX_SEQUENCE_LENGTH,
    )


def encode_day_payload(
    *,
    feature_values: np.ndarray,
    feature_mask: np.ndarray,
    tokens: np.ndarray,
    pc_contexts: np.ndarray,
    calendar_contexts: np.ndarray,
    gap_buckets: np.ndarray,
    seconds: np.ndarray,
) -> bytes:
    raw = b"".join(
        (
            _DAY_HEADER.pack(DAY_PAYLOAD_VERSION),
            np.asarray(feature_values, dtype="<f4").tobytes(),
            np.packbits(np.asarray(feature_mask, dtype=np.uint8), bitorder="little").tobytes(),
            np.asarray(tokens, dtype=np.uint8).tobytes(),
            np.asarray(pc_contexts, dtype=np.uint8).tobytes(),
            np.asarray(calendar_contexts, dtype=np.uint8).tobytes(),
            np.asarray(gap_buckets, dtype=np.uint8).tobytes(),
            np.asarray(seconds, dtype="<u4").tobytes(),
        )
    )
    return zlib.compress(raw, level=6)


@dataclass(frozen=True, slots=True)
class DecodedDay:
    feature_values: np.ndarray
    feature_mask: np.ndarray
    tokens: np.ndarray
    pc_contexts: np.ndarray
    calendar_contexts: np.ndarray
    gap_buckets: np.ndarray
    seconds: np.ndarray


def decode_day_payload(payload: bytes) -> DecodedDay:
    raw = memoryview(zlib.decompress(payload))
    if len(raw) < 1 or _DAY_HEADER.unpack(raw[:1])[0] != DAY_PAYLOAD_VERSION:
        raise ValueError("unsupported daily tensor payload")
    offset = 1

    def take(length: int) -> memoryview:
        nonlocal offset
        value = raw[offset : offset + length]
        if len(value) != length:
            raise ValueError("truncated daily tensor payload")
        offset += length
        return value

    feature_values = np.frombuffer(
        take(_DAY_FEATURE_BYTES),
        dtype="<f4",
    ).copy()
    packed_mask = np.frombuffer(take(_DAY_MASK_BYTES), dtype=np.uint8)
    feature_mask = np.unpackbits(
        packed_mask,
        bitorder="little",
    )[:FEATURE_DIMENSION].astype(bool)
    tokens = np.frombuffer(take(_DAY_SEQUENCE_BYTES), dtype=np.uint8).copy()
    pc_contexts = np.frombuffer(take(_DAY_SEQUENCE_BYTES), dtype=np.uint8).copy()
    calendars = np.frombuffer(take(_DAY_SEQUENCE_BYTES), dtype=np.uint8).copy()
    gaps = np.frombuffer(take(_DAY_SEQUENCE_BYTES), dtype=np.uint8).copy()
    seconds = np.frombuffer(take(_DAY_SECONDS_BYTES), dtype="<u4").copy()
    if offset != len(raw):
        raise ValueError("daily tensor payload has trailing bytes")
    return DecodedDay(
        feature_values=feature_values,
        feature_mask=feature_mask,
        tokens=tokens,
        pc_contexts=pc_contexts,
        calendar_contexts=calendars,
        gap_buckets=gaps,
        seconds=seconds,
    )


def _preprocessing_checksum(feature_config: Path, sequence_config: Path) -> str:
    implementation_files = [
        Path(__file__),
        Path(__file__).with_name("cert_data.py"),
        Path(__file__).with_name("preprocessing.py"),
        Path(__file__).with_name("temporal.py"),
        Path(__file__).with_name("contracts.py"),
    ]
    payload = {
        "schema_version": "cert-stream-preprocessing.v1",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
        "feature_config_sha256": hashlib.sha256(feature_config.read_bytes()).hexdigest(),
        "sequence_config_sha256": hashlib.sha256(sequence_config.read_bytes()).hexdigest(),
        "window": "[D-29,D]",
        "score_day": "D",
        "timestamp_basis": "CERT_LOCAL_WALL_CLOCK",
        "raw_content": "EXCLUDED",
        "pc_profiles": "TRAIN_ONLY_FROZEN",
        "temporal_reference": "TRAIN_PAST_ONLY_THEN_FROZEN",
        "event_order": ["timestamp", "source_rank", "event_uid"],
        "gap_ids": "PAD_PLUS_5_BUCKETS",
        "implementation_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in implementation_files
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _load_day_payloads(
    connection: sqlite3.Connection,
    day: date,
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for source, user_id, payload in connection.execute(
        """
        SELECT source,user_id,payload
        FROM source_days
        WHERE day=?
        ORDER BY user_id,source
        """,
        (day.isoformat(),),
    ):
        result[str(user_id)][str(source)] = _decompress_json(payload)
    return result


def materialize_daily_tensors(
    connection: sqlite3.Connection,
    *,
    raw_root: Path,
    feature_config: Path,
    sequence_config: Path,
    end_day: date = TEST_END,
) -> dict[str, object]:
    if end_day > TEST_END:
        raise ValueError(f"end_day cannot exceed locked CERT test end {TEST_END}")
    required_sources = set(SOURCE_FILES)
    complete_sources = {
        source
        for source in required_sources
        if metadata_get(connection, f"source_complete_{source}") is not None
    }
    if complete_sources != required_sources:
        raise ValueError(
            f"raw aggregate stage is incomplete: {sorted(required_sources - complete_sources)}"
        )
    signature_payload = {
        "end_day": end_day.isoformat(),
        "preprocessing_checksum": _preprocessing_checksum(
            feature_config,
            sequence_config,
        ),
        "sources": {
            source: metadata_get(connection, f"source_complete_{source}")
            for source in sorted(required_sources)
        },
    }
    signature = hashlib.sha256(
        json.dumps(
            signature_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if (
        metadata_get(connection, "daily_materialization_complete") == "true"
        and metadata_get(connection, "daily_materialization_signature") == signature
    ):
        total_rows, active_rows, truncated_rows = connection.execute(
            """
            SELECT COUNT(*),SUM(event_count > 0),SUM(truncated)
            FROM daily_tensors
            """
        ).fetchone()
        return {
            "status": "SKIPPED",
            "eligible_user_days": int(total_rows or 0),
            "active_user_days": int(active_rows or 0),
            "truncated_user_days": int(truncated_rows or 0),
            "start_day": (TRAIN_START - timedelta(days=WINDOW_DAYS - 1)).isoformat(),
            "end_day": end_day.isoformat(),
        }
    feature_names = load_feature_names(feature_config)
    directory = LdapDirectory.from_raw(raw_root)
    pc_profiles = load_store_pc_profiles(connection)
    if not pc_profiles:
        raise ValueError("Train-only PC profiles must be fitted before daily tensors")

    connection.execute("DELETE FROM daily_tensors")
    metadata_set(connection, "daily_materialization_complete", "false")
    connection.commit()
    histories: dict[str, UserFeatureHistory] = defaultdict(UserFeatureHistory)
    temporal = TemporalReferenceState()
    total_rows = 0
    active_rows = 0
    truncated_rows = 0
    materialization_start = TRAIN_START - timedelta(days=WINDOW_DAYS - 1)
    current = materialization_start
    while current <= end_day:
        temporal.expire(current)
        if current == TRAIN_END + timedelta(days=1):
            temporal.freeze()
        raw_by_user = _load_day_payloads(connection, current)
        rows: list[tuple[object, ...]] = []
        pending_temporal_updates: list[
            tuple[str, str, str, Sequence[int]]
        ] = []
        for user_id in sorted(directory.users_on(current)):
            profile = directory.profile(user_id, current)
            role_start = directory.role_start(user_id, current)
            if profile is None or role_start is None:
                continue
            role_epoch = f"{user_id}|{profile.role}|{role_start.isoformat()}"
            history = histories[user_id]
            history.expire(current)
            payloads = raw_by_user.get(user_id, {})
            all_seconds = [
                second
                for source in SOURCE_FILES
                for second in _payload_seconds(payloads.get(source))
            ]
            baseline = temporal.select(
                day=current,
                user_id=user_id,
                role=profile.role,
                role_epoch=role_epoch,
            )
            feature_values, feature_mask = _features_from_payloads(
                user_id=user_id,
                day=current,
                role_start=role_start,
                payloads=payloads,
                feature_names=feature_names,
                pc_profiles=pc_profiles,
                history=history,
                temporal_baseline=baseline,
            )
            (
                tokens,
                pc_contexts,
                calendar_contexts,
                gap_buckets,
                seconds,
                sequence_length,
                truncated,
            ) = _sequence_from_payloads(
                user_id=user_id,
                day=current,
                payloads=payloads,
                pc_profiles=pc_profiles,
            )
            event_count = len(all_seconds)
            rows.append(
                (
                    user_id,
                    current.isoformat(),
                    locked_split_for_day(current),
                    profile.role,
                    role_epoch,
                    event_count,
                    sequence_length,
                    int(feature_mask.sum()),
                    int(truncated),
                    encode_day_payload(
                        feature_values=feature_values,
                        feature_mask=feature_mask,
                        tokens=tokens,
                        pc_contexts=pc_contexts,
                        calendar_contexts=calendar_contexts,
                        gap_buckets=gap_buckets,
                        seconds=seconds,
                    ),
                )
            )
            history.event_days.append((current, event_count))
            host_hashes = set(
                payloads.get("HTTP", {}).get("data", {}).get("host_counts", {})
            )
            history.domain_days.append(
                (current, host_hashes, _payload_count(payloads.get("HTTP")))
            )
            external_hashes = set(
                payloads.get("EMAIL", {}).get("data", {}).get("external_hashes", [])
            )
            history.recipient_days.append((current, external_hashes))
            if event_count:
                history.last_active_day = current
                active_rows += 1
            if truncated:
                truncated_rows += 1
            if TRAIN_START <= current <= TRAIN_END:
                pending_temporal_updates.append(
                    (user_id, profile.role, role_epoch, all_seconds)
                )
        connection.executemany(
            """
            INSERT INTO daily_tensors(
                user_id,day,split,role,role_epoch,event_count,sequence_length,
                feature_observed_count,truncated,payload
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        connection.commit()
        for user_id, role, role_epoch, all_seconds in pending_temporal_updates:
            temporal.add_train_day(
                day=current,
                user_id=user_id,
                role=role,
                role_epoch=role_epoch,
                seconds=all_seconds,
            )
        total_rows += len(rows)
        metadata_set(connection, "daily_materialization_last_day", current.isoformat())
        connection.commit()
        current += timedelta(days=1)
    if end_day >= TRAIN_END and temporal.frozen_global is None:
        temporal.freeze()
    metadata_set(connection, "daily_materialization_complete", "true")
    metadata_set(connection, "daily_materialization_signature", signature)
    metadata_set(connection, "daily_materialization_end_day", end_day.isoformat())
    metadata_set(
        connection,
        "preprocessing_checksum",
        _preprocessing_checksum(feature_config, sequence_config),
    )
    connection.commit()
    return {
        "status": "BUILT",
        "eligible_user_days": total_rows,
        "active_user_days": active_rows,
        "truncated_user_days": truncated_rows,
        "start_day": materialization_start.isoformat(),
        "end_day": end_day.isoformat(),
    }


def fit_store_scaler(connection: sqlite3.Connection) -> RobustFeatureScaler:
    count = int(
        connection.execute(
            "SELECT COUNT(*) FROM daily_tensors WHERE split='TRAIN'"
        ).fetchone()[0]
    )
    if count == 0:
        raise ValueError("daily store has no Train user-days")
    values = np.zeros((count, FEATURE_DIMENSION), dtype=np.float32)
    mask = np.zeros((count, FEATURE_DIMENSION), dtype=bool)
    for index, (payload,) in enumerate(
        connection.execute(
            """
            SELECT payload FROM daily_tensors
            WHERE split='TRAIN'
            ORDER BY day,user_id
            """
        )
    ):
        decoded = decode_day_payload(payload)
        values[index] = decoded.feature_values
        mask[index] = decoded.feature_mask
    scaler = fit_robust_feature_scaler(values, mask)
    payload = {
        "schema_version": "robust-feature-scaler.v1",
        "fit_split": "TRAIN",
        "frozen_after_train": True,
        "location": scaler.location.tolist(),
        "scale": scaler.scale.tolist(),
        "support": scaler.support.tolist(),
        "checksum": scaler.checksum,
    }
    metadata_set(connection, "scaler", payload)
    metadata_set(connection, "scaler_checksum", scaler.checksum)
    connection.commit()
    return scaler


def load_store_scaler(connection: sqlite3.Connection) -> RobustFeatureScaler:
    raw = metadata_get(connection, "scaler")
    if raw is None:
        raise ValueError("store does not contain a Train-fitted scaler")
    payload = json.loads(raw)
    return RobustFeatureScaler(
        location=np.asarray(payload["location"], dtype=np.float64),
        scale=np.asarray(payload["scale"], dtype=np.float64),
        support=np.asarray(payload["support"], dtype=np.int64),
        checksum=str(payload["checksum"]),
    )


def store_readiness_report(connection: sqlite3.Connection) -> dict[str, object]:
    scaler = load_store_scaler(connection)
    train_user_days, train_users = connection.execute(
        """
        SELECT COUNT(*),COUNT(DISTINCT user_id)
        FROM daily_tensors
        WHERE split='TRAIN'
        """
    ).fetchone()
    sequence_days, sequence_users, transitions = connection.execute(
        """
        SELECT
            SUM(sequence_length > 0),
            COUNT(DISTINCT CASE WHEN sequence_length > 0 THEN user_id END),
            SUM(CASE WHEN sequence_length > 0 THEN sequence_length - 1 ELSE 0 END)
        FROM daily_tensors
        WHERE split='TRAIN'
        """
    ).fetchone()
    feature_coverage_ratio = float(np.mean(scaler.support >= 200))
    feature_ready = (
        int(train_users or 0) >= 200
        and int(train_user_days or 0) >= 10000
        and feature_coverage_ratio >= 0.9
    )
    sequence_ready = (
        int(sequence_users or 0) >= 200
        and int(sequence_days or 0) >= 10000
        and int(transitions or 0) >= 100000
    )
    return {
        "feature_global": {
            "ready": feature_ready,
            "users": int(train_users or 0),
            "user_days": int(train_user_days or 0),
            "feature_coverage_ratio_at_200": feature_coverage_ratio,
        },
        "sequence_global": {
            "ready": sequence_ready,
            "users": int(sequence_users or 0),
            "sequence_days": int(sequence_days or 0),
            "transitions": int(transitions or 0),
        },
        "at_least_one_branch_ready": feature_ready or sequence_ready,
    }


def build_stream_store(
    *,
    raw_root: Path,
    store_path: Path,
    feature_config: Path,
    sequence_config: Path,
    max_rows_per_source: int | None = None,
    end_day: date = TEST_END,
) -> dict[str, object]:
    raw_root = raw_root.resolve()
    connection = connect_store(store_path)
    try:
        initialize_store(connection)
        if metadata_get(connection, "schema_version") != STORE_SCHEMA_VERSION:
            raise ValueError("existing store has an incompatible schema")
        source_results = [
            materialize_source(
                connection,
                raw_root=raw_root,
                source=source,
                max_rows=max_rows_per_source,
            )
            for source in SOURCE_FILES
        ]
        pc_result = fit_store_pc_profiles(connection)
        daily_result = materialize_daily_tensors(
            connection,
            raw_root=raw_root,
            feature_config=feature_config,
            sequence_config=sequence_config,
            end_day=end_day,
        )
        scaler = (
            load_store_scaler(connection)
            if daily_result.get("status") == "SKIPPED"
            and metadata_get(connection, "scaler") is not None
            else fit_store_scaler(connection)
        )
        readiness = store_readiness_report(connection)
        metadata_set(connection, "raw_root", str(raw_root))
        metadata_set(connection, "raw_data_copied", "false")
        metadata_set(connection, "content_policy", "METADATA_ONLY")
        metadata_set(connection, "timestamp_basis", "CERT_LOCAL_WALL_CLOCK")
        metadata_set(connection, "feature_schema_version", FEATURE_SCHEMA_VERSION)
        metadata_set(connection, "sequence_schema_version", SEQUENCE_SCHEMA_VERSION)
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        manifest = {
            "schema_version": "cert-stream-store-manifest.v1",
            "store_schema_version": STORE_SCHEMA_VERSION,
            "raw_root": str(raw_root),
            "raw_data_copied": False,
            "content_selected": False,
            "max_rows_per_source": max_rows_per_source,
            "sources": source_results,
            "pc": pc_result,
            "daily": daily_result,
            "scaler_checksum": scaler.checksum,
            "readiness_preflight": readiness,
            "preprocessing_checksum": metadata_get(
                connection,
                "preprocessing_checksum",
            ),
            "store": str(store_path.resolve()),
        }
    finally:
        connection.close()
    manifest["store_size_bytes"] = store_path.stat().st_size
    manifest_path = store_path.with_suffix(store_path.suffix + ".manifest.json")
    atomic_write_json(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest
