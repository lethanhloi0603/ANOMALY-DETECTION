"""Versioned tensor contracts shared by preparation, training, and inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

FEATURE_DIMENSION = 128
MAX_SEQUENCE_LENGTH = 256
FEATURE_SCHEMA_VERSION = "feature128.v5"
SEQUENCE_SCHEMA_VERSION = "sequence7.v4"
SEQUENCE_TOKENS = (
    "PAD",
    "LOGON",
    "LOGOFF",
    "DEVICE_CONNECT",
    "DEVICE_DISCONNECT",
    "FILE",
    "HTTP",
    "EMAIL",
)
CALENDAR_CONTEXTS = ("PAD", "WEEKDAY", "WEEKEND")
PC_CONTEXTS = ("PAD", "OWN", "SHARED", "FOREIGN", "UNKNOWN")
GAP_BUCKETS = ("PAD", "0-1", "1-5", "5-30", "30-120", ">120")
WINDOW_DAYS = 30
CHECKSUM_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TRAIN_START = date(2010, 1, 2)
TRAIN_END = date(2010, 5, 31)
VALIDATION_START = date(2010, 6, 1)
VALIDATION_END = date(2010, 9, 30)
TEST_START = date(2010, 10, 1)
TEST_END = date(2011, 5, 17)


def locked_split_for_day(day: date) -> str:
    """Return the immutable CERT research split for an employee-day."""

    if TRAIN_START <= day <= TRAIN_END:
        return "TRAIN"
    if VALIDATION_START <= day <= VALIDATION_END:
        return "VALIDATION"
    if TEST_START <= day <= TEST_END:
        return "TEST"
    return "PRODUCTION"


@dataclass(frozen=True, slots=True)
class PreparedWindows:
    """A batch of fixed-length user-day windows.

    Shapes are ``[sample, day, feature]`` for feature tensors and
    ``[sample, day, event]`` for sequence tensors. Zero is reserved for PAD in
    every categorical side channel. Cyclic clock channels are zero on PAD and
    lie on the unit circle for observed events.
    """

    feature_values: np.ndarray
    feature_mask: np.ndarray
    tokens: np.ndarray
    token_mask: np.ndarray
    pc_contexts: np.ndarray
    calendar_contexts: np.ndarray
    gap_buckets: np.ndarray
    time_sin: np.ndarray
    time_cos: np.ndarray
    window_dates: np.ndarray
    eligible_day_mask: np.ndarray
    sample_user_ids: np.ndarray
    sample_end_days: np.ndarray
    sample_splits: np.ndarray
    feature_schema_version: np.ndarray
    sequence_schema_version: np.ndarray
    preprocessing_checksum: np.ndarray
    scaler_checksum: np.ndarray
    feature_value_space: np.ndarray

    def validate(self) -> None:
        if self.feature_values.ndim != 3:
            raise ValueError("feature_values must have shape [sample, day, feature]")
        if self.feature_values.shape[-1] != FEATURE_DIMENSION:
            raise ValueError(f"feature dimension must be {FEATURE_DIMENSION}")
        if self.feature_values.shape[1] != WINDOW_DAYS:
            raise ValueError(f"day axis must be the locked {WINDOW_DAYS}-day window")
        if self.feature_mask.shape != self.feature_values.shape:
            raise ValueError("feature_mask must match feature_values")
        expected_day_shape = self.feature_values.shape[:2]
        if self.window_dates.shape != expected_day_shape:
            raise ValueError("window_dates must have shape [sample, day]")
        if self.eligible_day_mask.shape != expected_day_shape:
            raise ValueError("eligible_day_mask must have shape [sample, day]")
        eligible = self.eligible_day_mask.astype(bool)
        if np.any(self.feature_mask.astype(bool) & ~eligible[:, :, None]):
            raise ValueError("feature observations cannot exist on an ineligible employee-day")

        expected_sequence_shape = self.tokens.shape
        if self.tokens.ndim != 3:
            raise ValueError("tokens must have shape [sample, day, event]")
        if self.tokens.shape[:2] != self.feature_values.shape[:2]:
            raise ValueError("feature and sequence sample/day axes must match")
        if self.tokens.shape[-1] != MAX_SEQUENCE_LENGTH:
            raise ValueError(f"event axis must equal locked max_len={MAX_SEQUENCE_LENGTH}")
        for name, value in (
            ("token_mask", self.token_mask),
            ("pc_contexts", self.pc_contexts),
            ("calendar_contexts", self.calendar_contexts),
            ("gap_buckets", self.gap_buckets),
            ("time_sin", self.time_sin),
            ("time_cos", self.time_cos),
        ):
            if value.shape != expected_sequence_shape:
                raise ValueError(f"{name} must match tokens")

        if np.any((self.tokens < 0) | (self.tokens >= len(SEQUENCE_TOKENS))):
            raise ValueError(f"tokens contain an ID outside {SEQUENCE_SCHEMA_VERSION}")
        for name, values, upper_bound in (
            ("pc_contexts", self.pc_contexts, len(PC_CONTEXTS)),
            ("calendar_contexts", self.calendar_contexts, len(CALENDAR_CONTEXTS)),
            ("gap_buckets", self.gap_buckets, len(GAP_BUCKETS)),
        ):
            if np.any((values < 0) | (values >= upper_bound)):
                raise ValueError(f"{name} contains an ID outside its locked vocabulary")
        if not np.array_equal(self.token_mask.astype(bool), self.tokens != 0):
            raise ValueError("token_mask must be true exactly for non-PAD tokens")
        if np.any(self.token_mask.astype(bool) & ~eligible[:, :, None]):
            raise ValueError("sequence events cannot exist on an ineligible employee-day")
        padding = ~self.token_mask.astype(bool)
        if any(
            np.any(values[padding] != 0)
            for values in (self.pc_contexts, self.calendar_contexts, self.gap_buckets)
        ):
            raise ValueError("all sequence side channels must be PAD where token_mask is false")
        for name, values in (("time_sin", self.time_sin), ("time_cos", self.time_cos)):
            if np.any(~np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
            if np.any((values < -1.0) | (values > 1.0)):
                raise ValueError(f"{name} must stay in [-1, 1]")
            if np.any(values[padding] != 0.0):
                raise ValueError(f"{name} must be zero where token_mask is false")
        observed = self.token_mask.astype(bool)
        cyclic_norm = np.square(self.time_sin[observed]) + np.square(self.time_cos[observed])
        if not np.allclose(cyclic_norm, 1.0, atol=1e-4):
            raise ValueError("time_sin/time_cos must lie on the unit circle for observed events")
        if np.any(~np.isfinite(self.feature_values[self.feature_mask.astype(bool)])):
            raise ValueError("observed feature values must be finite")
        if np.any(~eligible.any(axis=1)):
            raise ValueError("every sample must contain at least one eligible employee-day")

        sample_count = self.feature_values.shape[0]
        for name, values in (
            ("sample_user_ids", self.sample_user_ids),
            ("sample_end_days", self.sample_end_days),
            ("sample_splits", self.sample_splits),
        ):
            if values.shape != (sample_count,):
                raise ValueError(f"{name} must have shape [sample]")
        if any(not str(value).strip() for value in self.sample_user_ids):
            raise ValueError("sample_user_ids cannot contain empty values")
        for sample_index, raw_day in enumerate(self.sample_end_days):
            try:
                end_day = date.fromisoformat(str(raw_day))
            except ValueError as exc:
                raise ValueError(f"sample_end_days contains invalid ISO date: {raw_day!r}") from exc
            expected_dates = [
                (end_day - timedelta(days=WINDOW_DAYS - 1 - offset)).isoformat()
                for offset in range(WINDOW_DAYS)
            ]
            actual_dates = [str(value) for value in self.window_dates[sample_index]]
            if actual_dates != expected_dates:
                raise ValueError("window_dates must be the exact past-only range [D-29,D]")
        allowed_splits = {"TRAIN", "VALIDATION", "TEST", "PRODUCTION"}
        normalized_splits = {str(value).upper() for value in self.sample_splits}
        if not normalized_splits <= allowed_splits:
            raise ValueError("sample_splits contains an unsupported split")
        for raw_day, raw_split in zip(
            self.sample_end_days,
            self.sample_splits,
            strict=True,
        ):
            expected_split = locked_split_for_day(date.fromisoformat(str(raw_day)))
            if str(raw_split).upper() != expected_split:
                raise ValueError(
                    f"sample split {raw_split!r} conflicts with locked date split "
                    f"{expected_split!r} for {raw_day}"
                )

        scalar_contracts = (
            ("feature_schema_version", self.feature_schema_version, FEATURE_SCHEMA_VERSION),
            ("sequence_schema_version", self.sequence_schema_version, SEQUENCE_SCHEMA_VERSION),
            ("feature_value_space", self.feature_value_space, "robust_scaled"),
        )
        for name, value, expected in scalar_contracts:
            if value.shape != () or str(value.item()) != expected:
                raise ValueError(f"{name} must be scalar {expected!r}")
        for name, value in (
            ("preprocessing_checksum", self.preprocessing_checksum),
            ("scaler_checksum", self.scaler_checksum),
        ):
            if value.shape != () or not CHECKSUM_PATTERN.fullmatch(str(value.item())):
                raise ValueError(f"{name} must be a scalar lowercase SHA-256")
