"""Loading of model-ready NPZ windows; raw-log parsing stays outside the model."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from insider_ml.contracts import (
    FEATURE_DIMENSION,
    MAX_SEQUENCE_LENGTH,
    TRAIN_START,
    WINDOW_DAYS,
    PreparedWindows,
)
from insider_ml.stream_store import (
    DecodedDay,
    connect_store,
    decode_day_payload,
    load_store_scaler,
    metadata_get,
)

REQUIRED_ARRAYS = (
    "feature_values",
    "feature_mask",
    "tokens",
    "token_mask",
    "pc_contexts",
    "calendar_contexts",
    "gap_buckets",
    "time_sin",
    "time_cos",
    "window_dates",
    "eligible_day_mask",
    "sample_user_ids",
    "sample_end_days",
    "sample_splits",
    "feature_schema_version",
    "sequence_schema_version",
    "preprocessing_checksum",
    "scaler_checksum",
    "feature_value_space",
)


def load_prepared_windows(path: str | Path) -> PreparedWindows:
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_ARRAYS) - set(archive.files))
        if missing:
            raise ValueError(f"prepared window archive is missing arrays: {missing}")
        windows = PreparedWindows(**{name: archive[name] for name in REQUIRED_ARRAYS})
    windows.validate()
    return windows


class WindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, windows: PreparedWindows) -> None:
        windows.validate()
        self._windows = windows

    def __len__(self) -> int:
        return self._windows.feature_values.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data = self._windows
        return {
            "feature_values": torch.as_tensor(data.feature_values[index], dtype=torch.float32),
            "feature_mask": torch.as_tensor(data.feature_mask[index], dtype=torch.bool),
            "tokens": torch.as_tensor(data.tokens[index], dtype=torch.long),
            "token_mask": torch.as_tensor(data.token_mask[index], dtype=torch.bool),
            "pc_contexts": torch.as_tensor(data.pc_contexts[index], dtype=torch.long),
            "calendar_contexts": torch.as_tensor(
                data.calendar_contexts[index], dtype=torch.long
            ),
            "gap_buckets": torch.as_tensor(data.gap_buckets[index], dtype=torch.long),
            "time_sin": torch.as_tensor(data.time_sin[index], dtype=torch.float32),
            "time_cos": torch.as_tensor(data.time_cos[index], dtype=torch.float32),
            "eligible_day_mask": torch.as_tensor(
                data.eligible_day_mask[index], dtype=torch.bool
            ),
        }


@dataclass(frozen=True, slots=True)
class StoreSample:
    user_id: str
    day: str
    split: str
    role: str
    role_epoch: str
    event_count: int
    sequence_length: int
    feature_observed_count: int


class SQLiteWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Read past-only windows from a compact daily store with one-user caching."""

    def __init__(
        self,
        path: str | Path,
        *,
        split: str,
        max_samples: int | None = None,
        endpoint_policy: str = "ALL",
    ) -> None:
        self.path = Path(path).resolve()
        self.split = split.upper()
        connection = connect_store(self.path, read_only=True)
        try:
            if metadata_get(connection, "daily_materialization_complete") != "true":
                raise ValueError("daily tensor materialization is incomplete")
            self.preprocessing_checksum = metadata_get(
                connection,
                "preprocessing_checksum",
            )
            self.scaler_checksum = metadata_get(connection, "scaler_checksum")
            if self.preprocessing_checksum is None or self.scaler_checksum is None:
                raise ValueError("store is missing preprocessing/scaler checksums")
            self.scaler = load_store_scaler(connection)
            self.samples = [
                StoreSample(
                    user_id=str(row[0]),
                    day=str(row[1]),
                    split=str(row[2]),
                    role=str(row[3]),
                    role_epoch=str(row[4]),
                    event_count=int(row[5]),
                    sequence_length=int(row[6]),
                    feature_observed_count=int(row[7]),
                )
                for row in connection.execute(
                    """
                    SELECT
                        user_id,day,split,role,role_epoch,event_count,
                        sequence_length,feature_observed_count
                    FROM daily_tensors
                    WHERE split=?
                    ORDER BY user_id,day
                    """,
                    (self.split,),
                )
            ]
            policy = endpoint_policy.upper()
            if policy == "WEEKLY_TRAIN":
                if self.split != "TRAIN":
                    raise ValueError("WEEKLY_TRAIN is valid only for the Train split")
                self.samples = [
                    sample
                    for sample in self.samples
                    if (
                        date.fromisoformat(sample.day) - TRAIN_START
                    ).days
                    % 7
                    == int.from_bytes(
                        hashlib.sha256(sample.user_id.encode()).digest()[:4],
                        "little",
                    )
                    % 7
                ]
            elif policy != "ALL":
                raise ValueError(f"unsupported endpoint policy {endpoint_policy!r}")
            self.endpoint_policy = policy
            if max_samples is not None:
                if max_samples < 1:
                    raise ValueError("max_samples must be positive")
                self.samples = self.samples[:max_samples]
        finally:
            connection.close()
        if not self.samples:
            raise ValueError(f"store has no endpoint samples for split={self.split}")
        self._connection: sqlite3.Connection | None = None
        self._cached_user_id: str | None = None
        self._cached_user_days: dict[str, DecodedDay] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_connection"] = None
        state["_cached_user_id"] = None
        state["_cached_user_days"] = {}
        return state

    def __len__(self) -> int:
        return len(self.samples)

    def sample_metadata(self, index: int) -> StoreSample:
        return self.samples[index]

    def _ensure_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = connect_store(self.path, read_only=True)
        return self._connection

    def close(self) -> None:
        if getattr(self, "_connection", None) is not None:
            self._connection.close()
            self._connection = None
        self._cached_user_id = None
        self._cached_user_days = {}

    def __del__(self) -> None:
        self.close()

    def _load_user(self, user_id: str) -> None:
        if self._cached_user_id == user_id:
            return
        connection = self._ensure_connection()
        self._cached_user_days = {
            str(day): decode_day_payload(payload)
            for day, payload in connection.execute(
                """
                SELECT day,payload
                FROM daily_tensors
                WHERE user_id=?
                ORDER BY day
                """,
                (user_id,),
            )
        }
        self._cached_user_id = user_id

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        self._load_user(sample.user_id)
        end_day = date.fromisoformat(sample.day)
        feature_values = np.zeros(
            (WINDOW_DAYS, FEATURE_DIMENSION),
            dtype=np.float32,
        )
        feature_mask = np.zeros(
            (WINDOW_DAYS, FEATURE_DIMENSION),
            dtype=bool,
        )
        tokens = np.zeros(
            (WINDOW_DAYS, MAX_SEQUENCE_LENGTH),
            dtype=np.int64,
        )
        pc_contexts = np.zeros_like(tokens)
        calendar_contexts = np.zeros_like(tokens)
        gap_buckets = np.zeros_like(tokens)
        time_sin = np.zeros_like(tokens, dtype=np.float32)
        time_cos = np.zeros_like(tokens, dtype=np.float32)
        eligible_day_mask = np.zeros(WINDOW_DAYS, dtype=bool)
        for day_index in range(WINDOW_DAYS):
            current = end_day - timedelta(days=WINDOW_DAYS - 1 - day_index)
            decoded = self._cached_user_days.get(current.isoformat())
            if decoded is None:
                continue
            eligible_day_mask[day_index] = True
            feature_values[day_index] = decoded.feature_values
            feature_mask[day_index] = decoded.feature_mask
            tokens[day_index] = decoded.tokens
            pc_contexts[day_index] = decoded.pc_contexts
            calendar_contexts[day_index] = decoded.calendar_contexts
            gap_buckets[day_index] = decoded.gap_buckets
            observed = decoded.tokens != 0
            angles = (
                2.0
                * np.pi
                * decoded.seconds[observed].astype(np.float64)
                / 86400.0
            )
            time_sin[day_index, observed] = np.sin(angles)
            time_cos[day_index, observed] = np.cos(angles)
        feature_values = self.scaler.transform(feature_values, feature_mask)
        token_mask = tokens != 0
        return {
            "feature_values": torch.as_tensor(feature_values, dtype=torch.float32),
            "feature_mask": torch.as_tensor(feature_mask, dtype=torch.bool),
            "tokens": torch.as_tensor(tokens, dtype=torch.long),
            "token_mask": torch.as_tensor(token_mask, dtype=torch.bool),
            "pc_contexts": torch.as_tensor(pc_contexts, dtype=torch.long),
            "calendar_contexts": torch.as_tensor(
                calendar_contexts,
                dtype=torch.long,
            ),
            "gap_buckets": torch.as_tensor(gap_buckets, dtype=torch.long),
            "time_sin": torch.as_tensor(time_sin, dtype=torch.float32),
            "time_cos": torch.as_tensor(time_cos, dtype=torch.float32),
            "eligible_day_mask": torch.as_tensor(
                eligible_day_mask,
                dtype=torch.bool,
            ),
        }
