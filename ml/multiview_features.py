from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

CANONICAL_EVENT_COLUMNS = [
    "event_uid", "source_file", "original_id", "timestamp", "date", "user", "pc",
    "event_type", "object", "object_type", "hour", "is_after_hours", "is_weekend",
    "event_order", "source_payload", "role", "business_unit", "functional_unit",
    "department", "team", "supervisor"
]

TOKEN_VOCAB = [
    "PAD", "UNK", "DAY_START", "DAY_END",
    "LOGON", "LOGOFF", "LOGON_AFTER_HOURS", "LOGON_OTHER_PC",
    "DEVICE_CONNECT", "DEVICE_DISCONNECT", "DEVICE_CONNECT_AFTER_HOURS",
    "FILE_ACCESS", "FILE_ACCESS_AFTER_HOURS", "FILE_SENSITIVE_EXT", "FILE_BURST",
    "HTTP_ACCESS", "HTTP_JOB_SITE", "HTTP_WIKILEAKS", "HTTP_EXTERNAL_STORAGE", "HTTP_SUSPICIOUS_DOWNLOAD",
    "EMAIL_SEND", "EMAIL_EXTERNAL", "EMAIL_ATTACHMENT", "EMAIL_MASS", "EMAIL_BCC"
]
TOKEN_TO_ID = {t: i for i, t in enumerate(TOKEN_VOCAB)}
SOURCE_TO_ID = {"logon": 0, "device": 1, "file": 2, "http": 3, "email": 4, "unknown": 5}
COUNT_FEATURE_COLUMNS = [
    "total_events", "active_hours", "first_event_hour", "last_event_hour", "n_unique_pcs",
    "n_logon", "n_logoff", "n_logon_after_hours", "n_distinct_logon_pc",
    "n_device_connect", "n_device_disconnect", "n_device_after_hours", "has_device_use",
    "n_file_events", "n_unique_files", "n_file_after_hours", "sum_file_content_len", "mean_file_content_len", "n_sensitive_file_ext",
    "n_http_events", "n_unique_domains", "n_job_site_visits", "n_wikileaks_visits", "has_wikileaks_access", "n_http_after_hours",
    "n_email_sent", "sum_email_size", "mean_email_size", "total_recipient_count", "external_recipient_count", "bcc_count", "attachment_count", "mass_email_score",
    "has_after_hours_logon_and_device", "has_device_and_file_burst", "has_device_and_external_email", "has_job_site_and_device", "has_wikileaks_and_after_hours",
    "z_file_vs_user_30d", "z_usb_vs_user_30d", "z_email_external_vs_role_30d",
]
SEQ_FEATURE_COLUMNS = [
    "seq_len", "seq_unique_tokens", "seq_after_hour_tokens", "seq_mean_gap_min", "seq_max_gap_min",
    "seq_has_logon_after_hours_then_device", "seq_has_device_then_file", "seq_has_file_then_http", "seq_has_job_then_device",
    "seq_has_wikileaks", "seq_has_external_email", "seq_has_bcc", "seq_file_burst_flag"
] + [f"tok_{t}" for t in TOKEN_VOCAB[4:]]

def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()



def series_or_default(df: pd.DataFrame, col: str, default="") -> pd.Series:
    """Return df[col] as a Series, or a same-length default Series when the column is missing."""
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)

def str_series(df: pd.DataFrame, col: str, default="") -> pd.Series:
    return series_or_default(df, col, default).astype(str).str.strip().replace({"nan": ""})

def numeric_series(df: pd.DataFrame, col: str, default=0) -> pd.Series:
    return pd.to_numeric(series_or_default(df, col, default), errors="coerce").fillna(default)

def parse_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)

def is_after_hour(ts: pd.Timestamp) -> int:
    try:
        return int(ts.hour < 8 or ts.hour >= 18)
    except Exception:
        return 0

def domain_from_url(url: str) -> str:
    url = safe_str(url).lower()
    m = re.search(r"https?://([^/]+)", url)
    if m:
        return m.group(1)
    return url.split("/")[0] if url else ""

def count_recipients(value: str) -> int:
    if not isinstance(value, str) or not value.strip():
        return 0
    return len([p for p in re.split(r"[,;]", value) if p.strip()])

def count_external(value: str) -> int:
    if not isinstance(value, str) or not value.strip():
        return 0
    parts = [p.strip().lower() for p in re.split(r"[,;]", value) if p.strip()]
    return sum(1 for p in parts if "@dtaa" not in p)

def read_csv_limited(path: Path, max_rows: int) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing {path.name}, skipping.", flush=True)
        return pd.DataFrame()
    print(f"[READ] {path.name} | max rows = {max_rows:,}", flush=True)
    return pd.read_csv(path, nrows=max_rows, low_memory=False)

def load_ldap_context(ldap_dir: Optional[str | Path]) -> pd.DataFrame:
    if not ldap_dir:
        return pd.DataFrame(columns=["user", "valid_month", "role", "business_unit", "functional_unit", "department", "team", "supervisor"])
    p = Path(ldap_dir)
    if not p.exists():
        return pd.DataFrame(columns=["user", "valid_month", "role", "business_unit", "functional_unit", "department", "team", "supervisor"])
    frames = []
    for f in sorted(p.glob("*.csv")):
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:
            print(f"[WARN] Cannot read LDAP {f}: {e}", flush=True)
            continue
        rename = {"user_id": "user"}
        df = df.rename(columns=rename)
        for col in ["user", "role", "business_unit", "functional_unit", "department", "team", "supervisor"]:
            if col not in df.columns:
                df[col] = ""
        m = re.search(r"(\d{4})[-_](\d{2})", f.stem)
        valid_month = f"{m.group(1)}-{m.group(2)}" if m else "0000-00"
        df["valid_month"] = valid_month
        frames.append(df[["user", "valid_month", "role", "business_unit", "functional_unit", "department", "team", "supervisor"]])
    if not frames:
        return pd.DataFrame(columns=["user", "valid_month", "role", "business_unit", "functional_unit", "department", "team", "supervisor"])
    out = pd.concat(frames, ignore_index=True)
    out["user"] = out["user"].astype(str).str.strip()
    for col in ["role", "business_unit", "functional_unit", "department", "team", "supervisor"]:
        out[col] = out[col].astype(str).str.strip().replace({"nan": ""})
    return out.drop_duplicates(["user", "valid_month"], keep="last")

def attach_role_context(events: pd.DataFrame, role_context: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    if role_context.empty:
        for col in ["role", "business_unit", "functional_unit", "department", "team", "supervisor"]:
            events[col] = "UNKNOWN"
        return events
    events["valid_month"] = pd.to_datetime(events["date"]).dt.strftime("%Y-%m")
    merged = events.merge(role_context, on=["user", "valid_month"], how="left")
    missing = merged["role"].isna() | (merged["role"].astype(str).str.strip() == "")
    if missing.any():
        latest = role_context.sort_values("valid_month").drop_duplicates("user", keep="last")
        latest = latest.drop(columns=["valid_month"])
        fallback = events.loc[missing, ["user"]].merge(latest, on="user", how="left")
        for col in ["role", "business_unit", "functional_unit", "department", "team", "supervisor"]:
            merged.loc[missing, col] = fallback[col].values
    for col in ["role", "business_unit", "functional_unit", "department", "team", "supervisor"]:
        merged[col] = merged[col].fillna("UNKNOWN").replace("", "UNKNOWN")
    return merged.drop(columns=["valid_month"], errors="ignore")

def normalize_logon(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = pd.DataFrame()
    out["original_id"] = str_series(df, "id")
    out["timestamp"] = parse_ts(series_or_default(df, "date"))
    out["user"] = str_series(df, "user")
    out["pc"] = str_series(df, "pc")
    act = str_series(df, "activity")
    out["event_type"] = np.where(act.str.lower().str.contains("logoff"), "LOGOFF", "LOGON")
    out["object"] = out["pc"]
    out["object_type"] = "pc"
    out["source_file"] = "logon"
    out["source_payload"] = act.map(lambda x: json.dumps({"raw_activity": x}))
    return finalize_events(out)

def normalize_device(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = pd.DataFrame()
    out["original_id"] = str_series(df, "id")
    out["timestamp"] = parse_ts(series_or_default(df, "date"))
    out["user"] = str_series(df, "user")
    out["pc"] = str_series(df, "pc")
    act = str_series(df, "activity")
    out["event_type"] = np.where(act.str.lower().str.contains("disconnect"), "DEVICE_DISCONNECT", "DEVICE_CONNECT")
    out["object"] = out["pc"]
    out["object_type"] = "device"
    out["source_file"] = "device"
    out["source_payload"] = act.map(lambda x: json.dumps({"raw_activity": x}))
    return finalize_events(out)

def normalize_file(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = pd.DataFrame()
    out["original_id"] = str_series(df, "id")
    out["timestamp"] = parse_ts(series_or_default(df, "date"))
    out["user"] = str_series(df, "user")
    out["pc"] = str_series(df, "pc")
    filename = str_series(df, "filename")
    content = str_series(df, "content")
    out["event_type"] = "FILE_ACCESS"
    out["object"] = filename
    out["object_type"] = "file"
    out["source_file"] = "file"
    out["source_payload"] = [json.dumps({"filename": f, "content_len": len(c), "file_ext": Path(f).suffix.lower()}) for f, c in zip(filename, content)]
    return finalize_events(out)

def normalize_http(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = pd.DataFrame()
    out["original_id"] = str_series(df, "id")
    out["timestamp"] = parse_ts(series_or_default(df, "date"))
    out["user"] = str_series(df, "user")
    out["pc"] = str_series(df, "pc")
    url = str_series(df, "url")
    content = str_series(df, "content")
    out["event_type"] = "HTTP_ACCESS"
    out["object"] = url
    out["object_type"] = "url"
    out["source_file"] = "http"
    out["source_payload"] = [json.dumps({"url": u, "domain": domain_from_url(u), "content_len": len(c)}) for u, c in zip(url, content)]
    return finalize_events(out)

def normalize_email(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = pd.DataFrame()
    out["original_id"] = str_series(df, "id")
    out["timestamp"] = parse_ts(series_or_default(df, "date"))
    out["user"] = str_series(df, "user")
    out["pc"] = str_series(df, "pc")
    to = str_series(df, "to")
    cc = str_series(df, "cc")
    bcc = str_series(df, "bcc")
    size = numeric_series(df, "size", 0)
    att = numeric_series(df, "attachment_count", 0)
    out["event_type"] = "EMAIL_SEND"
    out["object"] = to
    out["object_type"] = "email"
    out["source_file"] = "email"
    out["source_payload"] = [json.dumps({"to": a, "cc": b, "bcc": c, "size": float(s), "attachment_count": int(t), "external_count": count_external(a)+count_external(b)+count_external(c)}) for a,b,c,s,t in zip(to,cc,bcc,size,att)]
    return finalize_events(out)

def finalize_events(out: pd.DataFrame) -> pd.DataFrame:
    out = out.dropna(subset=["timestamp"])
    out = out[out["user"].astype(str).str.strip() != ""]
    out["date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    out["hour"] = out["timestamp"].dt.hour.astype(int)
    out["is_after_hours"] = ((out["hour"] < 8) | (out["hour"] >= 18)).astype(int)
    out["is_weekend"] = (out["timestamp"].dt.dayofweek >= 5).astype(int)
    out["event_uid"] = out["source_file"].astype(str) + ":" + out["original_id"].astype(str)
    out["event_order"] = 0
    return out[CANONICAL_EVENT_COLUMNS[:15]]

def build_unified_event_log(cert_dir: str | Path, max_rows_per_file: int, ldap_dir: Optional[str | Path] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cert_dir = Path(cert_dir)
    frames = [
        normalize_logon(read_csv_limited(cert_dir / "logon.csv", max_rows_per_file)),
        normalize_device(read_csv_limited(cert_dir / "device.csv", max_rows_per_file)),
        normalize_file(read_csv_limited(cert_dir / "file.csv", max_rows_per_file)),
        normalize_email(read_csv_limited(cert_dir / "email.csv", max_rows_per_file)),
        normalize_http(read_csv_limited(cert_dir / "http.csv", max_rows_per_file)),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FileNotFoundError(f"No CERT logs found under {cert_dir}")
    events = pd.concat(frames, ignore_index=True)
    events = events.sort_values(["user", "timestamp", "source_file", "original_id"]).reset_index(drop=True)
    events["event_order"] = events.groupby("user").cumcount().astype(int)
    role_context = load_ldap_context(ldap_dir)
    events = attach_role_context(events, role_context)
    return events[CANONICAL_EVENT_COLUMNS], role_context

def payload_get(payload: str, key: str, default=None):
    try:
        return json.loads(payload).get(key, default)
    except Exception:
        return default

def build_user_day_features(events: pd.DataFrame) -> pd.DataFrame:
    e = events.copy()
    e["timestamp"] = pd.to_datetime(e["timestamp"], utc=True, errors="coerce")
    e["is_logon"] = (e["event_type"] == "LOGON").astype(int)
    e["is_logoff"] = (e["event_type"] == "LOGOFF").astype(int)
    e["is_device_connect"] = (e["event_type"] == "DEVICE_CONNECT").astype(int)
    e["is_device_disconnect"] = (e["event_type"] == "DEVICE_DISCONNECT").astype(int)
    e["is_file"] = (e["event_type"] == "FILE_ACCESS").astype(int)
    e["is_http"] = (e["event_type"] == "HTTP_ACCESS").astype(int)
    e["is_email"] = (e["event_type"] == "EMAIL_SEND").astype(int)
    e["domain"] = e.apply(lambda r: payload_get(r["source_payload"], "domain", domain_from_url(r["object"])), axis=1)
    e["is_job_site"] = e["object"].astype(str).str.lower().str.contains("job|career|linkedin|monster", regex=True).astype(int)
    e["is_wikileaks"] = e["object"].astype(str).str.lower().str.contains("wikileaks").astype(int)
    e["file_ext"] = e["source_payload"].map(lambda p: payload_get(p, "file_ext", ""))
    e["content_len"] = e["source_payload"].map(lambda p: payload_get(p, "content_len", 0)).astype(float)
    e["is_sensitive_ext"] = e["file_ext"].astype(str).str.lower().isin([".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".pdf", ".zip", ".7z", ".rar"]).astype(int)
    e["email_size"] = e["source_payload"].map(lambda p: payload_get(p, "size", 0)).astype(float)
    e["attachment"] = e["source_payload"].map(lambda p: payload_get(p, "attachment_count", 0)).astype(float)
    e["external_count"] = e["source_payload"].map(lambda p: payload_get(p, "external_count", 0)).astype(float)
    e["recipient_count"] = e.apply(lambda r: count_recipients(payload_get(r["source_payload"], "to", "")) + count_recipients(payload_get(r["source_payload"], "cc", "")) + count_recipients(payload_get(r["source_payload"], "bcc", "")), axis=1)
    e["bcc_count_raw"] = e["source_payload"].map(lambda p: count_recipients(payload_get(p, "bcc", "")))
    grp = e.groupby(["user", "date"], as_index=False)
    f = grp.agg(
        total_events=("event_uid", "count"), active_hours=("hour", pd.Series.nunique), first_event_hour=("hour", "min"), last_event_hour=("hour", "max"), n_unique_pcs=("pc", pd.Series.nunique),
        n_logon=("is_logon", "sum"), n_logoff=("is_logoff", "sum"), n_logon_after_hours=("is_after_hours", lambda s: 0), n_distinct_logon_pc=("pc", pd.Series.nunique),
        n_device_connect=("is_device_connect", "sum"), n_device_disconnect=("is_device_disconnect", "sum"), n_device_after_hours=("is_after_hours", lambda s: 0),
        n_file_events=("is_file", "sum"), n_unique_files=("object", pd.Series.nunique), n_file_after_hours=("is_after_hours", lambda s: 0), sum_file_content_len=("content_len", "sum"), mean_file_content_len=("content_len", "mean"), n_sensitive_file_ext=("is_sensitive_ext", "sum"),
        n_http_events=("is_http", "sum"), n_unique_domains=("domain", pd.Series.nunique), n_job_site_visits=("is_job_site", "sum"), n_wikileaks_visits=("is_wikileaks", "sum"), n_http_after_hours=("is_after_hours", lambda s: 0),
        n_email_sent=("is_email", "sum"), sum_email_size=("email_size", "sum"), mean_email_size=("email_size", "mean"), total_recipient_count=("recipient_count", "sum"), external_recipient_count=("external_count", "sum"), bcc_count=("bcc_count_raw", "sum"), attachment_count=("attachment", "sum"),
        role=("role", "last"), business_unit=("business_unit", "last"), functional_unit=("functional_unit", "last"), department=("department", "last"), team=("team", "last")
    )
    # Fix conditional after-hours per event type.
    def sum_cond(mask_col):
        return e[e[mask_col] == 1].groupby(["user", "date"])["is_after_hours"].sum()
    idx = f.set_index(["user", "date"]).index
    for name, mask in [("n_logon_after_hours", "is_logon"), ("n_device_after_hours", "is_device_connect"), ("n_file_after_hours", "is_file"), ("n_http_after_hours", "is_http")]:
        f[name] = sum_cond(mask).reindex(idx).fillna(0).values
    f["has_device_use"] = (f["n_device_connect"] > 0).astype(int)
    f["has_wikileaks_access"] = (f["n_wikileaks_visits"] > 0).astype(int)
    f["mass_email_score"] = f["total_recipient_count"] + 2 * f["bcc_count"] + 2 * f["external_recipient_count"]
    file_burst = f.groupby("user")["n_file_events"].transform(lambda s: s.quantile(0.9))
    f["has_after_hours_logon_and_device"] = ((f["n_logon_after_hours"] > 0) & (f["n_device_connect"] > 0)).astype(int)
    f["has_device_and_file_burst"] = ((f["n_device_connect"] > 0) & (f["n_file_events"] >= file_burst) & (f["n_file_events"] > 0)).astype(int)
    f["has_device_and_external_email"] = ((f["n_device_connect"] > 0) & (f["external_recipient_count"] > 0)).astype(int)
    f["has_job_site_and_device"] = ((f["n_job_site_visits"] > 0) & (f["n_device_connect"] > 0)).astype(int)
    f["has_wikileaks_and_after_hours"] = ((f["n_wikileaks_visits"] > 0) & (f["n_logon_after_hours"] > 0)).astype(int)
    f = f.sort_values(["user", "date"])
    for col, out in [("n_file_events", "z_file_vs_user_30d"), ("n_device_connect", "z_usb_vs_user_30d")]:
        mean = f.groupby("user")[col].transform(lambda s: s.shift(1).rolling(30, min_periods=5).mean())
        std = f.groupby("user")[col].transform(lambda s: s.shift(1).rolling(30, min_periods=5).std())
        f[out] = ((f[col] - mean) / std.replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0)
    role_mean = f.groupby(["role", "date"])["external_recipient_count"].transform("mean")
    role_std = f.groupby(["role", "date"])["external_recipient_count"].transform("std").replace(0, np.nan)
    f["z_email_external_vs_role_30d"] = ((f["external_recipient_count"] - role_mean) / role_std).replace([np.inf, -np.inf], 0).fillna(0)
    for col in COUNT_FEATURE_COLUMNS:
        if col not in f.columns:
            f[col] = 0
    f[COUNT_FEATURE_COLUMNS] = f[COUNT_FEATURE_COLUMNS].fillna(0)
    return f[["user", "date", "role", "business_unit", "functional_unit", "department", "team"] + COUNT_FEATURE_COLUMNS]

def token_for_event(row: pd.Series) -> str:
    et = row["event_type"]
    after = int(row.get("is_after_hours", 0)) == 1
    obj = safe_str(row.get("object", "")).lower()
    payload = safe_str(row.get("source_payload", ""))
    if et == "LOGON": return "LOGON_AFTER_HOURS" if after else "LOGON"
    if et == "LOGOFF": return "LOGOFF"
    if et == "DEVICE_CONNECT": return "DEVICE_CONNECT_AFTER_HOURS" if after else "DEVICE_CONNECT"
    if et == "DEVICE_DISCONNECT": return "DEVICE_DISCONNECT"
    if et == "FILE_ACCESS":
        ext = safe_str(payload_get(payload, "file_ext", "")).lower()
        if ext in [".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".pdf", ".zip"]: return "FILE_SENSITIVE_EXT"
        return "FILE_ACCESS_AFTER_HOURS" if after else "FILE_ACCESS"
    if et == "HTTP_ACCESS":
        if "wikileaks" in obj: return "HTTP_WIKILEAKS"
        if re.search(r"job|career|linkedin|monster", obj): return "HTTP_JOB_SITE"
        if re.search(r"dropbox|drive|mega|upload", obj): return "HTTP_EXTERNAL_STORAGE"
        return "HTTP_ACCESS"
    if et == "EMAIL_SEND":
        ext_count = payload_get(payload, "external_count", 0)
        att = payload_get(payload, "attachment_count", 0)
        bcc = count_recipients(payload_get(payload, "bcc", ""))
        if bcc > 0: return "EMAIL_BCC"
        if ext_count and float(ext_count) > 0: return "EMAIL_EXTERNAL"
        if att and float(att) > 0: return "EMAIL_ATTACHMENT"
        return "EMAIL_SEND"
    return "UNK"

def build_user_day_sequences(events: pd.DataFrame) -> pd.DataFrame:
    e = events.copy().sort_values(["user", "date", "timestamp", "event_order"])
    e["timestamp"] = pd.to_datetime(e["timestamp"], utc=True, errors="coerce")
    e["token"] = e.apply(token_for_event, axis=1)
    rows = []
    for (user, date), g in e.groupby(["user", "date"]):
        g = g.sort_values("timestamp")
        tokens = ["DAY_START"] + g["token"].tolist() + ["DAY_END"]
        token_ids = [TOKEN_TO_ID.get(t, TOKEN_TO_ID["UNK"]) for t in tokens]
        sources = ["unknown"] + g["source_file"].astype(str).tolist() + ["unknown"]
        source_ids = [SOURCE_TO_ID.get(s, SOURCE_TO_ID["unknown"]) for s in sources]
        ts = g["timestamp"].tolist()
        gaps = [0]
        for i in range(1, len(ts)):
            gaps.append(max(0, (ts[i] - ts[i - 1]).total_seconds() / 60))
        gaps = [0] + gaps + [0]
        seq = tokens
        def has_order(a, b):
            try: return seq.index(a) < seq.index(b)
            except ValueError: return False
        tok_counts = {f"tok_{t}": seq.count(t) for t in TOKEN_VOCAB[4:]}
        rows.append({
            "user": user, "date": date,
            "event_sequence": json.dumps(token_ids),
            "source_sequence": json.dumps(source_ids),
            "time_gap_sequence": json.dumps(gaps),
            "seq_len": len(tokens) - 2,
            "seq_unique_tokens": len(set(tokens)),
            "seq_after_hour_tokens": sum(1 for t in tokens if "AFTER_HOURS" in t),
            "seq_mean_gap_min": float(np.mean(gaps)) if gaps else 0,
            "seq_max_gap_min": float(np.max(gaps)) if gaps else 0,
            "seq_has_logon_after_hours_then_device": int(has_order("LOGON_AFTER_HOURS", "DEVICE_CONNECT")),
            "seq_has_device_then_file": int(any(t.startswith("DEVICE_CONNECT") for t in seq) and any(t.startswith("FILE") for t in seq)),
            "seq_has_file_then_http": int(any(t.startswith("FILE") for t in seq) and any(t.startswith("HTTP") for t in seq)),
            "seq_has_job_then_device": int(has_order("HTTP_JOB_SITE", "DEVICE_CONNECT")),
            "seq_has_wikileaks": int("HTTP_WIKILEAKS" in seq),
            "seq_has_external_email": int("EMAIL_EXTERNAL" in seq),
            "seq_has_bcc": int("EMAIL_BCC" in seq),
            "seq_file_burst_flag": int(sum(1 for t in seq if t.startswith("FILE")) >= 20),
            **tok_counts,
            "sequence_flags": json.dumps({"has_wikileaks": int("HTTP_WIKILEAKS" in seq), "has_external_email": int("EMAIL_EXTERNAL" in seq)})
        })
    out = pd.DataFrame(rows)
    for col in SEQ_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0
    return out

def build_user_day_multiview(features: pd.DataFrame, sequences: pd.DataFrame) -> pd.DataFrame:
    mv = features.merge(sequences, on=["user", "date"], how="left")
    for col in SEQ_FEATURE_COLUMNS:
        if col not in mv.columns:
            mv[col] = 0
    mv[SEQ_FEATURE_COLUMNS] = mv[SEQ_FEATURE_COLUMNS].fillna(0)
    mv["event_sequence"] = mv.get("event_sequence", "[]").fillna("[]")
    mv["source_sequence"] = mv.get("source_sequence", "[]").fillna("[]")
    mv["time_gap_sequence"] = mv.get("time_gap_sequence", "[]").fillna("[]")
    return mv.sort_values(["user", "date"]).reset_index(drop=True)

def read_events_from_sqlite(db_path: str, user_id: Optional[str] = None, role_context: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    where = ""
    params = []
    if user_id:
        where = "WHERE UserId = ?"
        params.append(user_id)
    query = f"""
    SELECT COALESCE(SourceEventId, lower(hex(randomblob(16)))) AS original_id,
           TimestampUtc AS timestamp, UserId AS user, PcId AS pc, EventType AS source_file,
           Activity AS activity, Url AS url, FileName AS filename, EmailTo AS email_to,
           EmailCc AS email_cc, EmailBcc AS email_bcc, EmailFrom AS email_from,
           Size AS size, AttachmentCount AS attachment_count, Content AS content
    FROM RawLogs {where} ORDER BY UserId, TimestampUtc
    """
    with sqlite3.connect(db_path) as conn:
        raw = pd.read_sql_query(query, conn, params=params)
    return normalize_runtime_rawlogs(raw, role_context=role_context)

def normalize_runtime_rawlogs(raw: pd.DataFrame, role_context: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    frames = []
    for src in ["logon", "device", "file", "email", "http"]:
        part = raw[raw["source_file"].astype(str).str.lower().str.contains(src, na=False)].copy()
        if part.empty: continue
        if src == "logon":
            part = part.rename(columns={"activity": "activity", "original_id": "id", "timestamp": "date"})
            frames.append(normalize_logon(part))
        elif src == "device":
            part = part.rename(columns={"activity": "activity", "original_id": "id", "timestamp": "date"})
            frames.append(normalize_device(part))
        elif src == "file":
            part = part.rename(columns={"filename": "filename", "original_id": "id", "timestamp": "date"})
            frames.append(normalize_file(part.assign(content=part.get("content", ""))))
        elif src == "email":
            part = part.rename(columns={"email_to": "to", "email_cc": "cc", "email_bcc": "bcc", "email_from": "from", "original_id": "id", "timestamp": "date"})
            frames.append(normalize_email(part))
        elif src == "http":
            part = part.rename(columns={"url": "url", "original_id": "id", "timestamp": "date"})
            frames.append(normalize_http(part.assign(content=part.get("content", ""))))
    if not frames:
        return pd.DataFrame(columns=CANONICAL_EVENT_COLUMNS)
    events = pd.concat(frames, ignore_index=True).sort_values(["user", "timestamp"]).reset_index(drop=True)
    events["event_order"] = events.groupby("user").cumcount().astype(int)
    if role_context is None:
        role_context = pd.DataFrame()
    return attach_role_context(events, role_context)[CANONICAL_EVENT_COLUMNS]

def multiview_from_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    features = build_user_day_features(events)
    sequences = build_user_day_sequences(events)
    return build_user_day_multiview(features, sequences)

def make_windows_from_multiview(df: pd.DataFrame, feature_cols: list[str], window_size: int, allow_padding=True):
    windows, metas = [], []
    if df.empty:
        return np.empty((0, window_size, len(feature_cols)), dtype=np.float32), []
    df = df.sort_values(["user", "date"]).copy()
    for user, g in df.groupby("user"):
        g = g.sort_values("date")
        values = g[feature_cols].astype(float).to_numpy(dtype=np.float32)
        dates = g["date"].astype(str).tolist()
        roles = g.get("role", pd.Series(["UNKNOWN"]*len(g))).astype(str).tolist()
        deps = g.get("department", pd.Series(["UNKNOWN"]*len(g))).astype(str).tolist()
        if len(values) < window_size:
            if allow_padding:
                pad = np.zeros((window_size - len(values), values.shape[1]), dtype=np.float32)
                windows.append(np.vstack([pad, values]))
                metas.append({"user": user, "date": dates[-1], "role": roles[-1], "department": deps[-1]})
            continue
        for start in range(0, len(values) - window_size + 1):
            end = start + window_size
            windows.append(values[start:end])
            metas.append({"user": user, "date": dates[end-1], "role": roles[end-1], "department": deps[end-1]})
    if not windows:
        return np.empty((0, window_size, len(feature_cols)), dtype=np.float32), []
    return np.stack(windows).astype(np.float32), metas
