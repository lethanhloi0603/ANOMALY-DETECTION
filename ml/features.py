from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "id", "date", "user", "pc", "event_type", "activity", "url", "filename",
    "to", "cc", "bcc", "from", "size", "attachment_count", "content",
]

FEATURE_COLUMNS = [
    "logon_count", "logoff_count", "device_connect_count", "device_disconnect_count",
    "file_copy_count", "email_count", "email_external_count", "email_attachment_total",
    "email_size_total", "http_count", "after_hour_count", "unique_pc_count", "unique_url_count",
    "event_logon_ratio", "event_device_ratio", "event_file_ratio", "event_email_ratio", "event_http_ratio",
]

DEVIATION_BASE_COLUMNS = [
    "logon_count", "device_connect_count", "file_copy_count", "email_count", "email_external_count", "http_count", "after_hour_count",
]


def normalize_event_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {
        "Id": "id", "SourceEventId": "id", "TimestampUtc": "date", "UserId": "user",
        "PcId": "pc", "EventType": "event_type", "Activity": "activity", "Url": "url",
        "FileName": "filename", "EmailTo": "to", "EmailCc": "cc", "EmailBcc": "bcc",
        "EmailFrom": "from", "Size": "size", "AttachmentCount": "attachment_count", "Content": "content",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[CANONICAL_COLUMNS]
    df["user"] = df["user"].astype(str).str.strip()
    df["pc"] = df["pc"].astype(str).str.strip()
    df["event_type"] = df["event_type"].astype(str).str.lower().str.strip()
    df["activity"] = df["activity"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date", "user"])
    df = df[df["user"] != ""]
    df = df.sort_values(["user", "date"])
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0)
    df["attachment_count"] = pd.to_numeric(df["attachment_count"], errors="coerce").fillna(0)
    return df


def read_events_from_sqlite(db_path: str, user_id: Optional[str] = None) -> pd.DataFrame:
    if not Path(db_path).exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")
    where = "WHERE UserId = ?" if user_id else ""
    params = [user_id] if user_id else []
    query = f"""
    SELECT COALESCE(SourceEventId, lower(hex(randomblob(16)))) AS SourceEventId,
           TimestampUtc, UserId, PcId, EventType, Activity, Url, FileName,
           EmailTo, EmailCc, EmailBcc, EmailFrom, Size, AttachmentCount, Content
    FROM RawLogs
    {where}
    ORDER BY UserId, TimestampUtc
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return normalize_event_frame(df)


def is_external_email(value: str) -> bool:
    if not isinstance(value, str) or value.strip() == "":
        return False
    parts = [p.strip().lower() for p in value.replace(";", ",").split(",") if p.strip()]
    return any("@dtaa" not in p for p in parts)


def build_daily_features(events: pd.DataFrame, feature_columns: Optional[list[str]] = None) -> pd.DataFrame:
    events = normalize_event_frame(events)
    if events.empty:
        cols = ["user", "day"] + (feature_columns or FEATURE_COLUMNS)
        return pd.DataFrame(columns=cols)

    events["day"] = events["date"].dt.floor("D")
    events["hour"] = events["date"].dt.hour
    events["is_after_hour"] = ((events["hour"] < 8) | (events["hour"] >= 18)).astype(int)
    et = events["event_type"].fillna("").str.lower()
    act = events["activity"].fillna("").str.lower()
    events["is_logon"] = ((et == "logon") & act.str.contains("logon", na=False)).astype(int)
    events["is_logoff"] = ((et == "logon") & act.str.contains("logoff", na=False)).astype(int)
    events["is_device_connect"] = ((et == "device") & act.str.contains("connect", na=False) & ~act.str.contains("disconnect", na=False)).astype(int)
    events["is_device_disconnect"] = ((et == "device") & act.str.contains("disconnect", na=False)).astype(int)
    events["is_file"] = (et == "file").astype(int)
    events["is_email"] = (et == "email").astype(int)
    events["is_http"] = (et == "http").astype(int)
    events["is_external_email"] = events.apply(lambda r: int(is_external_email(str(r.get("to", ""))) or is_external_email(str(r.get("cc", ""))) or is_external_email(str(r.get("bcc", "")))), axis=1)

    grouped = events.groupby(["user", "day"], as_index=False).agg(
        logon_count=("is_logon", "sum"), logoff_count=("is_logoff", "sum"),
        device_connect_count=("is_device_connect", "sum"), device_disconnect_count=("is_device_disconnect", "sum"),
        file_copy_count=("is_file", "sum"), email_count=("is_email", "sum"),
        email_external_count=("is_external_email", "sum"), email_attachment_total=("attachment_count", "sum"),
        email_size_total=("size", "sum"), http_count=("is_http", "sum"), after_hour_count=("is_after_hour", "sum"),
        unique_pc_count=("pc", pd.Series.nunique), unique_url_count=("url", pd.Series.nunique),
    )

    count_cols = ["logon_count", "device_connect_count", "file_copy_count", "email_count", "http_count"]
    total = grouped[count_cols].sum(axis=1).replace(0, 1)
    grouped["event_logon_ratio"] = grouped["logon_count"] / total
    grouped["event_device_ratio"] = grouped["device_connect_count"] / total
    grouped["event_file_ratio"] = grouped["file_copy_count"] / total
    grouped["event_email_ratio"] = grouped["email_count"] / total
    grouped["event_http_ratio"] = grouped["http_count"] / total

    filled = []
    for user, user_df in grouped.groupby("user"):
        user_df = user_df.sort_values("day")
        idx = pd.date_range(user_df["day"].min(), user_df["day"].max(), freq="D", tz="UTC")
        user_df = user_df.set_index("day").reindex(idx)
        user_df["user"] = user
        user_df.index.name = "day"
        user_df = user_df.reset_index()
        for col in FEATURE_COLUMNS:
            if col not in user_df.columns:
                user_df[col] = 0
        user_df[FEATURE_COLUMNS] = user_df[FEATURE_COLUMNS].fillna(0)
        filled.append(user_df[["user", "day"] + FEATURE_COLUMNS])
    daily = pd.concat(filled, ignore_index=True).sort_values(["user", "day"]).reset_index(drop=True)

    for col in DEVIATION_BASE_COLUMNS:
        mean = daily.groupby("user")[col].transform(lambda s: s.shift(1).rolling(14, min_periods=3).mean())
        std = daily.groupby("user")[col].transform(lambda s: s.shift(1).rolling(14, min_periods=3).std())
        daily[f"{col}_dev"] = ((daily[col] - mean) / std.replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0)

    final_cols = feature_columns or [c for c in daily.columns if c not in ["user", "day"]]
    for col in final_cols:
        if col not in daily.columns:
            daily[col] = 0
    return daily[["user", "day"] + final_cols]


def make_windows(daily: pd.DataFrame, window_size: int = 30, feature_columns: Optional[list[str]] = None, allow_padding: bool = True):
    if daily.empty:
        cols = feature_columns or FEATURE_COLUMNS + [f"{c}_dev" for c in DEVIATION_BASE_COLUMNS]
        return np.empty((0, window_size, len(cols)), dtype=np.float32), cols
    if feature_columns is None:
        feature_columns = [c for c in daily.columns if c not in ["user", "day"]]
    windows = []
    for _, user_df in daily.sort_values(["user", "day"]).groupby("user"):
        values = user_df[feature_columns].astype(float).to_numpy(dtype=np.float32)
        if len(values) < window_size:
            if allow_padding and len(values) > 0:
                pad = np.zeros((window_size - len(values), values.shape[1]), dtype=np.float32)
                windows.append(np.vstack([pad, values]))
            continue
        for start in range(0, len(values) - window_size + 1):
            windows.append(values[start:start + window_size])
    if not windows:
        return np.empty((0, window_size, len(feature_columns)), dtype=np.float32), feature_columns
    return np.stack(windows).astype(np.float32), feature_columns


def latest_user_window(events: pd.DataFrame, window_size: int, feature_columns: list[str]) -> np.ndarray:
    daily = build_daily_features(events, feature_columns=feature_columns)
    x, _ = make_windows(daily, window_size=window_size, feature_columns=feature_columns, allow_padding=True)
    if len(x) == 0:
        return np.zeros((1, window_size, len(feature_columns)), dtype=np.float32)
    return x[-1:]


def save_feature_columns(path: str | Path, columns: Iterable[str]) -> None:
    Path(path).write_text(json.dumps(list(columns), indent=2), encoding="utf-8")


def load_feature_columns(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
