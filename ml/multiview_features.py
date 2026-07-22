from __future__ import annotations

import json
import re
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_FEATURE_RULES = {
    "internal_email_domains": ["dtaa.com"],
    "missing_email_domain_is_external": False,
    "after_hours": {
        "enabled": False,
        "start_hour": 8,
        "end_hour": 18,
        "include_weekends": False,
    },
    "job_site_keywords": ["job", "career", "linkedin", "monster"],
    "wikileaks_keywords": ["wikileaks"],
    "external_storage_keywords": ["dropbox", "drive.google", "mega.nz", "upload"],
    "sensitive_file_extensions": [
        ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".pdf", ".zip", ".7z", ".rar"
    ],
    "file_burst": {"user_quantile": 0.9, "minimum_events": 20},
    "mass_email": {
        "recipient_weight": 1.0,
        "bcc_weight": 2.0,
        "external_weight": 2.0,
        "token_threshold": 20.0,
    },
    "max_events_per_day": 256,
}
FEATURE_RULES = deepcopy(DEFAULT_FEATURE_RULES)


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def configure_feature_rules(path: Optional[str | Path] = None) -> dict:
    """Load all heuristic rules from one auditable file.

    The default file lives next to this module. Callers may pass a different
    experiment-specific file, but rules must never be silently expanded in code.
    """
    global FEATURE_RULES
    rules_path = Path(path) if path else Path(__file__).with_name("feature_rules.json")
    overrides = {}
    if rules_path.exists():
        overrides = json.loads(rules_path.read_text(encoding="utf-8"))
    FEATURE_RULES = _deep_merge(DEFAULT_FEATURE_RULES, overrides)
    return deepcopy(FEATURE_RULES)


configure_feature_rules()

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
        config = FEATURE_RULES["after_hours"]
        if not config.get("enabled", False):
            return 0
        if config.get("include_weekends", False) and ts.dayofweek >= 5:
            return 1
        return int(ts.hour < int(config["start_hour"]) or ts.hour >= int(config["end_hour"]))
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

def _email_domain(address: str) -> str:
    address = safe_str(address).lower()
    if "@" not in address:
        return ""
    return address.rsplit("@", 1)[1].strip(" >")


def count_external(value: str) -> int:
    if not isinstance(value, str) or not value.strip():
        return 0
    parts = [p.strip().lower() for p in re.split(r"[,;]", value) if p.strip()]
    internal = {str(d).strip().lower() for d in FEATURE_RULES["internal_email_domains"]}
    missing_is_external = bool(FEATURE_RULES.get("missing_email_domain_is_external", False))
    external = 0
    for part in parts:
        domain = _email_domain(part)
        if not domain:
            external += int(missing_is_external)
        elif domain not in internal:
            external += 1
    return external


def contains_configured_keyword(value: str, rule_name: str) -> bool:
    text = safe_str(value).lower()
    return any(str(keyword).lower() in text for keyword in FEATURE_RULES.get(rule_name, []))

def read_csv_limited(path: Path, max_rows: int) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing {path.name}, skipping.", flush=True)
        return pd.DataFrame()
    if max_rows <= 0:
        print(f"[READ] {path.name} | full data", flush=True)
        return pd.read_csv(path, low_memory=False)
    print(f"[READ] {path.name} | first rows = {max_rows:,}", flush=True)
    return pd.read_csv(path, nrows=max_rows, low_memory=False)


def _month_keys(date_series: pd.Series, user_series: pd.Series) -> pd.Series:
    dates = date_series.astype(str)
    # CERT dates use MM/DD/YYYY HH:MM:SS; fall back to parser for unexpected rows.
    month = dates.str.slice(6, 10) + "-" + dates.str.slice(0, 2)
    invalid = ~month.str.fullmatch(r"\d{4}-\d{2}", na=False)
    if invalid.any():
        month.loc[invalid] = pd.to_datetime(dates.loc[invalid], errors="coerce").dt.strftime("%Y-%m")
    return user_series.astype(str).str.strip() + "|" + month.fillna("UNKNOWN")


def _collect_user_month_counts(paths: list[Path], chunk_size: int = 500_000) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for path in paths:
        source_counts: dict[str, int] = {}
        print(f"[AUDIT] counting user-month groups in {path.name}", flush=True)
        for chunk in pd.read_csv(path, usecols=["date", "user"], chunksize=chunk_size, low_memory=False):
            for key, count in _month_keys(chunk["date"], chunk["user"]).value_counts().items():
                source_counts[str(key)] = source_counts.get(str(key), 0) + int(count)
        counts[path.name] = source_counts
    return counts


def _forced_insider_keys(insiders_file: Optional[str | Path], available_keys: set[str]) -> set[str]:
    if not insiders_file or not Path(insiders_file).exists():
        return set()
    insiders = pd.read_csv(insiders_file, low_memory=False)
    if "dataset" in insiders.columns:
        insiders = insiders[insiders["dataset"].astype(str).str.strip() == "4.2"]
    forced = set()
    for row in insiders.itertuples(index=False):
        start = pd.to_datetime(row.start, errors="coerce")
        end = pd.to_datetime(row.end, errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        for month in pd.period_range(start=start, end=end, freq="M"):
            key = f"{row.user}|{month.strftime('%Y-%m')}"
            if key in available_keys:
                forced.add(key)
    return forced


def _select_time_user_groups(
    counts_by_source: dict[str, dict[str, int]],
    max_rows_per_file: int,
    random_seed: int,
    insiders_file: Optional[str | Path],
) -> set[str]:
    available_keys = set().union(*(set(source_counts) for source_counts in counts_by_source.values()))
    if not available_keys:
        return set()
    months = sorted({key.rsplit("|", 1)[-1] for key in available_keys})
    per_month_budget = max(1, max_rows_per_file // max(len(months), 1))
    selected: set[str] = set()
    usage = {source: {month: 0 for month in months} for source in counts_by_source}

    forced = _forced_insider_keys(insiders_file, available_keys)
    ordered_by_month: dict[str, list[str]] = {month: [] for month in months}
    rng = np.random.default_rng(random_seed)
    for month in months:
        candidates = sorted(key for key in available_keys if key.endswith(f"|{month}"))
        rng.shuffle(candidates)
        ordered_by_month[month] = candidates

    for key in list(sorted(forced)) + [key for month in months for key in ordered_by_month[month]]:
        if key in selected:
            continue
        month = key.rsplit("|", 1)[-1]
        additions = {source: source_counts.get(key, 0) for source, source_counts in counts_by_source.items()}
        if key not in forced and any(usage[source][month] + count > per_month_budget for source, count in additions.items()):
            continue
        selected.add(key)
        for source, count in additions.items():
            usage[source][month] += count
    print(f"[SAMPLE] selected {len(selected):,} complete user-month groups across {len(months)} months", flush=True)
    return selected


def read_time_user_stratified(
    paths: list[Path],
    max_rows_per_file: int,
    random_seed: int = 42,
    insiders_file: Optional[str | Path] = None,
    chunk_size: int = 500_000,
) -> dict[str, pd.DataFrame]:
    """Two-pass sampling that keeps complete user-month timelines across sources."""
    counts = _collect_user_month_counts(paths, chunk_size=chunk_size)
    selected = _select_time_user_groups(counts, max_rows_per_file, random_seed, insiders_file)
    if not selected:
        raise ValueError("Time-user stratified sampling selected no user-month groups")
    outputs = {}
    for path in paths:
        pieces = []
        print(f"[READ] {path.name} | selected complete user-month groups", flush=True)
        for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
            mask = _month_keys(chunk["date"], chunk["user"]).isin(selected)
            if mask.any():
                pieces.append(chunk.loc[mask].copy())
        outputs[path.name] = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return outputs

def load_ldap_context(ldap_path: Optional[str | Path]) -> pd.DataFrame:
    if not ldap_path:
        return pd.DataFrame(columns=["user", "valid_month", "role", "business_unit", "functional_unit", "department", "team", "supervisor"])
    p = Path(ldap_path)
    if not p.exists():
        return pd.DataFrame(columns=["user", "valid_month", "role", "business_unit", "functional_unit", "department", "team", "supervisor"])
    if p.is_file():
        files = [p]
    else:
        files = sorted(f for f in p.glob("*.csv") if re.fullmatch(r"\d{4}[-_]\d{2}", f.stem))
    frames = []
    for f in files:
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
        if "valid_month" not in df.columns:
            m = re.fullmatch(r"(\d{4})[-_](\d{2})", f.stem)
            if not m:
                print(f"[WARN] LDAP file has no valid_month and no YYYY-MM filename: {f}", flush=True)
                continue
            df["valid_month"] = f"{m.group(1)}-{m.group(2)}"
        else:
            df["valid_month"] = df["valid_month"].astype(str).str.slice(0, 7)
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
    context_cols = ["role", "business_unit", "functional_unit", "department", "team", "supervisor"]
    events["valid_month"] = pd.to_datetime(events["date"], errors="coerce").dt.strftime("%Y-%m")
    merged = events.merge(role_context, on=["user", "valid_month"], how="left")
    missing = merged["role"].isna() | (merged["role"].astype(str).str.strip() == "")
    # Past-only as-of fallback. Never attach a future LDAP snapshot to an older event.
    if missing.any():
        context = role_context.copy()
        context["valid_month_ts"] = pd.to_datetime(context["valid_month"] + "-01", errors="coerce")
        event_month = pd.to_datetime(merged["valid_month"] + "-01", errors="coerce")
        for user, indices in merged.loc[missing].groupby("user").groups.items():
            user_context = context[context["user"] == user].dropna(subset=["valid_month_ts"]).sort_values("valid_month_ts")
            if user_context.empty:
                continue
            months = user_context["valid_month_ts"].to_numpy(dtype="datetime64[ns]")
            positions = np.searchsorted(months, event_month.loc[indices].to_numpy(dtype="datetime64[ns]"), side="right") - 1
            valid = positions >= 0
            valid_indices = np.asarray(list(indices))[valid]
            if len(valid_indices) == 0:
                continue
            selected = user_context.iloc[positions[valid]]
            for col in context_cols:
                merged.loc[valid_indices, col] = selected[col].to_numpy()
    for col in context_cols:
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
    attachment_column = "attachment_count" if "attachment_count" in df.columns else "attachments"
    att = numeric_series(df, attachment_column, 0)
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
    out["is_after_hours"] = out["timestamp"].map(is_after_hour).astype(int)
    out["is_weekend"] = (out["timestamp"].dt.dayofweek >= 5).astype(int)
    out["event_uid"] = out["source_file"].astype(str) + ":" + out["original_id"].astype(str)
    out["event_order"] = 0
    return out[CANONICAL_EVENT_COLUMNS[:15]]

def build_unified_event_log(
    cert_dir: str | Path,
    max_rows_per_file: int,
    ldap_dir: Optional[str | Path] = None,
    sampling_mode: str = "time-user-stratified",
    random_seed: int = 42,
    insiders_file: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cert_dir = Path(cert_dir)
    paths = [cert_dir / name for name in ["logon.csv", "device.csv", "file.csv", "email.csv", "http.csv"]]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CERT log files: {missing}")
    if sampling_mode == "time-user-stratified":
        raw = read_time_user_stratified(
            paths,
            max_rows_per_file=max_rows_per_file,
            random_seed=random_seed,
            insiders_file=insiders_file,
        )
    elif sampling_mode == "full":
        raw = {path.name: read_csv_limited(path, 0) for path in paths}
    elif sampling_mode == "head":
        raw = {path.name: read_csv_limited(path, max_rows_per_file) for path in paths}
    else:
        raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")
    frames = [
        normalize_logon(raw["logon.csv"]),
        normalize_device(raw["device.csv"]),
        normalize_file(raw["file.csv"]),
        normalize_email(raw["email.csv"]),
        normalize_http(raw["http.csv"]),
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
    e["is_job_site"] = e["object"].map(lambda value: contains_configured_keyword(value, "job_site_keywords")).astype(int)
    e["is_wikileaks"] = e["object"].map(lambda value: contains_configured_keyword(value, "wikileaks_keywords")).astype(int)
    e["file_ext"] = e["source_payload"].map(lambda p: payload_get(p, "file_ext", ""))
    e["content_len"] = e["source_payload"].map(lambda p: payload_get(p, "content_len", 0)).astype(float)
    sensitive_extensions = {str(ext).lower() for ext in FEATURE_RULES["sensitive_file_extensions"]}
    e["is_sensitive_ext"] = e["file_ext"].astype(str).str.lower().isin(sensitive_extensions).astype(int)
    e["email_size"] = e["source_payload"].map(lambda p: payload_get(p, "size", 0)).astype(float)
    e["attachment"] = e["source_payload"].map(lambda p: payload_get(p, "attachment_count", 0)).astype(float)
    e["external_count"] = e["source_payload"].map(lambda p: payload_get(p, "external_count", 0)).astype(float)
    e["recipient_count"] = e.apply(lambda r: count_recipients(payload_get(r["source_payload"], "to", "")) + count_recipients(payload_get(r["source_payload"], "cc", "")) + count_recipients(payload_get(r["source_payload"], "bcc", "")), axis=1)
    e["bcc_count_raw"] = e["source_payload"].map(lambda p: count_recipients(payload_get(p, "bcc", "")))
    e["logon_pc"] = e["pc"].where(e["is_logon"] == 1)
    e["file_object"] = e["object"].where(e["is_file"] == 1)
    e["http_domain"] = e["domain"].where(e["is_http"] == 1)
    grp = e.groupby(["user", "date"], as_index=False)
    f = grp.agg(
        total_events=("event_uid", "count"), active_hours=("hour", pd.Series.nunique), first_event_hour=("hour", "min"), last_event_hour=("hour", "max"), n_unique_pcs=("pc", pd.Series.nunique),
        n_logon=("is_logon", "sum"), n_logoff=("is_logoff", "sum"), n_logon_after_hours=("is_after_hours", lambda s: 0), n_distinct_logon_pc=("logon_pc", pd.Series.nunique),
        n_device_connect=("is_device_connect", "sum"), n_device_disconnect=("is_device_disconnect", "sum"), n_device_after_hours=("is_after_hours", lambda s: 0),
        n_file_events=("is_file", "sum"), n_unique_files=("file_object", pd.Series.nunique), n_file_after_hours=("is_after_hours", lambda s: 0), sum_file_content_len=("content_len", "sum"), mean_file_content_len=("content_len", "mean"), n_sensitive_file_ext=("is_sensitive_ext", "sum"),
        n_http_events=("is_http", "sum"), n_unique_domains=("http_domain", pd.Series.nunique), n_job_site_visits=("is_job_site", "sum"), n_wikileaks_visits=("is_wikileaks", "sum"), n_http_after_hours=("is_after_hours", lambda s: 0),
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
    mass_email = FEATURE_RULES["mass_email"]
    f["mass_email_score"] = (
        float(mass_email["recipient_weight"]) * f["total_recipient_count"]
        + float(mass_email["bcc_weight"]) * f["bcc_count"]
        + float(mass_email["external_weight"]) * f["external_recipient_count"]
    )
    burst_rules = FEATURE_RULES["file_burst"]
    file_burst = f.groupby("user")["n_file_events"].transform(
        lambda s: max(float(s.quantile(float(burst_rules["user_quantile"]))), float(burst_rules["minimum_events"]))
    )
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
    # Past-only 30-calendar-day role baseline; never include the current day in its own baseline.
    f["date_ts"] = pd.to_datetime(f["date"], errors="coerce")
    role_day = (
        f.groupby(["role", "date_ts"], as_index=False)["external_recipient_count"]
        .mean()
        .sort_values(["role", "date_ts"])
    )
    baseline_rows = []
    for role, group in role_day.groupby("role"):
        group = group.sort_values("date_ts").set_index("date_ts")
        history = group["external_recipient_count"].shift(1).rolling("30D", min_periods=5)
        part = group.reset_index()[["date_ts"]]
        part["role"] = role
        part["role_mean_30d"] = history.mean().to_numpy()
        part["role_std_30d"] = history.std().to_numpy()
        baseline_rows.append(part)
    if baseline_rows:
        role_baseline = pd.concat(baseline_rows, ignore_index=True)
        f = f.merge(role_baseline, on=["role", "date_ts"], how="left")
        f["z_email_external_vs_role_30d"] = (
            (f["external_recipient_count"] - f["role_mean_30d"])
            / f["role_std_30d"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], 0).fillna(0)
    else:
        f["z_email_external_vs_role_30d"] = 0
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
        if ext in {str(value).lower() for value in FEATURE_RULES["sensitive_file_extensions"]}: return "FILE_SENSITIVE_EXT"
        return "FILE_ACCESS_AFTER_HOURS" if after else "FILE_ACCESS"
    if et == "HTTP_ACCESS":
        if contains_configured_keyword(obj, "wikileaks_keywords"): return "HTTP_WIKILEAKS"
        if contains_configured_keyword(obj, "job_site_keywords"): return "HTTP_JOB_SITE"
        if contains_configured_keyword(obj, "external_storage_keywords"): return "HTTP_EXTERNAL_STORAGE"
        return "HTTP_ACCESS"
    if et == "EMAIL_SEND":
        ext_count = payload_get(payload, "external_count", 0)
        att = payload_get(payload, "attachment_count", 0)
        bcc = count_recipients(payload_get(payload, "bcc", ""))
        recipients = (
            count_recipients(payload_get(payload, "to", ""))
            + count_recipients(payload_get(payload, "cc", ""))
            + bcc
        )
        mass_email = FEATURE_RULES["mass_email"]
        mass_score = (
            float(mass_email["recipient_weight"]) * recipients
            + float(mass_email["bcc_weight"]) * bcc
            + float(mass_email["external_weight"]) * float(ext_count or 0)
        )
        if mass_score >= float(mass_email["token_threshold"]): return "EMAIL_MASS"
        if bcc > 0: return "EMAIL_BCC"
        if ext_count and float(ext_count) > 0: return "EMAIL_EXTERNAL"
        if att and float(att) > 0: return "EMAIL_ATTACHMENT"
        return "EMAIL_SEND"
    return "UNK"

def build_user_day_sequences(events: pd.DataFrame) -> pd.DataFrame:
    e = events.copy().sort_values(["user", "date", "timestamp", "event_order"])
    e["timestamp"] = pd.to_datetime(e["timestamp"], utc=True, errors="coerce")
    e["token"] = e.apply(token_for_event, axis=1)
    burst_rules = FEATURE_RULES["file_burst"]
    daily_file_counts = (
        e.assign(is_file=(e["event_type"] == "FILE_ACCESS").astype(int))
        .groupby(["user", "date"])["is_file"]
        .sum()
    )
    burst_thresholds = daily_file_counts.groupby(level="user").transform(
        lambda values: max(
            float(values.quantile(float(burst_rules["user_quantile"]))),
            float(burst_rules["minimum_events"]),
        )
    )
    rows = []
    for (user, date), g in e.groupby(["user", "date"]):
        g = g.sort_values("timestamp")
        file_burst_flag = int(daily_file_counts.loc[(user, date)] >= burst_thresholds.loc[(user, date)])
        summary_tokens = ["FILE_BURST"] if file_burst_flag else []
        tokens = ["DAY_START"] + g["token"].tolist() + summary_tokens + ["DAY_END"]
        token_ids = [TOKEN_TO_ID.get(t, TOKEN_TO_ID["UNK"]) for t in tokens]
        sources = ["unknown"] + g["source_file"].astype(str).tolist() + ["unknown"] * (len(summary_tokens) + 1)
        source_ids = [SOURCE_TO_ID.get(s, SOURCE_TO_ID["unknown"]) for s in sources]
        ts = g["timestamp"].tolist()
        gaps = [0]
        for i in range(1, len(ts)):
            gaps.append(max(0, (ts[i] - ts[i - 1]).total_seconds() / 60))
        gaps = [0] + gaps + [0] * (len(summary_tokens) + 1)
        seq = tokens
        def first_prefix(prefix: str) -> Optional[int]:
            return next((i for i, token in enumerate(seq) if token.startswith(prefix)), None)

        def has_order(a: str, b: str) -> bool:
            a_index = first_prefix(a)
            b_index = first_prefix(b)
            return a_index is not None and b_index is not None and a_index < b_index
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
            "seq_has_device_then_file": int(has_order("DEVICE_CONNECT", "FILE")),
            "seq_has_file_then_http": int(has_order("FILE", "HTTP")),
            "seq_has_job_then_device": int(has_order("HTTP_JOB_SITE", "DEVICE_CONNECT")),
            "seq_has_wikileaks": int("HTTP_WIKILEAKS" in seq),
            "seq_has_external_email": int("EMAIL_EXTERNAL" in seq),
            "seq_has_bcc": int("EMAIL_BCC" in seq),
            "seq_file_burst_flag": file_burst_flag,
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


def attach_cert_labels(multiview: pd.DataFrame, insiders_path: Optional[str | Path], dataset_version: str = "4.2") -> pd.DataFrame:
    """Attach CERT labels for evaluation only; callers must not use them for fitting.

    A day is positive when it overlaps a matching insider interval. Scenario is
    retained as metadata so scenario-wise recall can be reported.
    """
    result = multiview.copy()
    result["label_day"] = 0
    result["scenario"] = ""
    if not insiders_path or not Path(insiders_path).exists() or result.empty:
        return result
    insiders = pd.read_csv(insiders_path, low_memory=False)
    if "dataset" in insiders.columns:
        insiders = insiders[insiders["dataset"].astype(str).str.strip() == str(dataset_version)]
    if insiders.empty:
        return result
    insiders["start_date"] = pd.to_datetime(insiders["start"], errors="coerce").dt.normalize()
    insiders["end_date"] = pd.to_datetime(insiders["end"], errors="coerce").dt.normalize()
    result_date = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    for row in insiders.itertuples(index=False):
        mask = (
            (result["user"].astype(str) == str(row.user))
            & (result_date >= row.start_date)
            & (result_date <= row.end_date)
        )
        result.loc[mask, "label_day"] = 1
        result.loc[mask, "scenario"] = str(row.scenario)
    return result


def calendarize_multiview(df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per calendar day for every user's observed timeline.

    Missing days get zero behavioral values, explicit empty day sequences and
    past-only forward-filled context. This makes a 30-day window mean 30
    consecutive calendar days rather than 30 active rows.
    """
    if df.empty:
        return df.copy()
    value_columns = list(dict.fromkeys(COUNT_FEATURE_COLUMNS + SEQ_FEATURE_COLUMNS))
    context_columns = ["role", "business_unit", "functional_unit", "department", "team"]
    sequence_columns = ["event_sequence", "source_sequence", "time_gap_sequence", "sequence_flags"]
    label_columns = [column for column in ["label_day", "scenario"] if column in df.columns]
    pieces = []
    for user, user_rows in df.groupby("user", sort=False):
        group = user_rows.copy()
        group["_date_ts"] = pd.to_datetime(group["date"], errors="coerce").dt.normalize()
        group = group.dropna(subset=["_date_ts"]).sort_values("_date_ts").drop_duplicates("_date_ts", keep="last")
        if group.empty:
            continue
        full_dates = pd.date_range(group["_date_ts"].min(), group["_date_ts"].max(), freq="D")
        group = group.set_index("_date_ts").reindex(full_dates)
        group["user"] = user
        group["date"] = group.index.strftime("%Y-%m-%d")
        for column in value_columns:
            if column not in group.columns:
                group[column] = 0
            group[column] = pd.to_numeric(group[column], errors="coerce").fillna(0)
        for column in context_columns:
            if column not in group.columns:
                group[column] = "UNKNOWN"
            group[column] = group[column].ffill().fillna("UNKNOWN").replace("", "UNKNOWN")
        defaults = {
            "event_sequence": json.dumps([TOKEN_TO_ID["DAY_START"], TOKEN_TO_ID["DAY_END"]]),
            "source_sequence": json.dumps([SOURCE_TO_ID["unknown"], SOURCE_TO_ID["unknown"]]),
            "time_gap_sequence": json.dumps([0, 0]),
            "sequence_flags": "{}",
        }
        for column in sequence_columns:
            if column not in group.columns:
                group[column] = defaults[column]
            group[column] = group[column].fillna(defaults[column])
        for column in label_columns:
            if column == "label_day":
                group[column] = pd.to_numeric(group[column], errors="coerce").fillna(0).astype(int)
            else:
                group[column] = group[column].fillna("").astype(str)
        pieces.append(group.reset_index(drop=True))
    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True).sort_values(["user", "date"]).reset_index(drop=True)


def assign_temporal_splits(df: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15) -> pd.DataFrame:
    """Assign chronological user-day splits without ever randomizing windows."""
    result = df.copy().sort_values(["user", "date"]).reset_index(drop=True)
    result["split"] = "test"
    for _, indices in result.groupby("user", sort=False).groups.items():
        ordered = list(indices)
        count = len(ordered)
        train_end = max(1, min(count, int(np.floor(count * train_fraction))))
        validation_end = max(train_end + 1, int(np.floor(count * (train_fraction + validation_fraction)))) if count >= 3 else count
        validation_end = min(validation_end, count)
        result.loc[ordered[:train_end], "split"] = "train"
        result.loc[ordered[train_end:validation_end], "split"] = "validation"
        result.loc[ordered[validation_end:], "split"] = "test"
    return result


def _parse_sequence(value, default: list) -> list:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else list(default)
    except Exception:
        return list(default)


def _truncate_head_tail(values: list, max_length: int) -> list:
    if len(values) <= max_length:
        return values
    head = max_length // 2
    return values[:head] + values[-(max_length - head):]


def make_multiview_day_inputs(
    df: pd.DataFrame,
    count_columns: list[str],
    window_size: int,
    max_events_per_day: Optional[int] = None,
    allow_padding: bool = True,
):
    """Build compact day-level arrays plus lazy calendar-window slices.

    The previous training path materialized every overlapping window. A single
    calendar day was therefore duplicated up to ``window_size`` times, and the
    token/source arrays were stored as int64. This representation stores every
    day once (uint16 tokens, uint8 sources, float32 numeric values) and records
    only ``(start, end)`` slices. Callers can materialize one batch at a time.
    """
    max_events = int(max_events_per_day or FEATURE_RULES["max_events_per_day"])
    empty_count = np.empty((0, len(count_columns)), dtype=np.float32)
    empty_tokens = np.empty((0, max_events), dtype=np.uint16)
    empty_sources = np.empty((0, max_events), dtype=np.uint8)
    empty_gaps = np.empty((0, max_events), dtype=np.float32)
    if df.empty:
        return empty_count, empty_tokens, empty_sources, empty_gaps, [], []

    calendar = calendarize_multiview(df)
    for column in count_columns:
        if column not in calendar.columns:
            calendar[column] = 0
        calendar[column] = pd.to_numeric(calendar[column], errors="coerce").fillna(0)

    count_days, token_days, source_days, gap_days = [], [], [], []
    window_slices: list[tuple[int, int]] = []
    metas: list[dict] = []
    day_offset = 0

    for user, group in calendar.groupby("user", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        count_values = group[count_columns].to_numpy(dtype=np.float32)
        day_tokens = np.zeros((len(group), max_events), dtype=np.uint16)
        day_sources = np.full(
            (len(group), max_events),
            SOURCE_TO_ID["unknown"],
            dtype=np.uint8,
        )
        day_gaps = np.zeros((len(group), max_events), dtype=np.float32)
        original_lengths = np.zeros(len(group), dtype=np.int32)

        for row_index, row in group.iterrows():
            raw_tokens = _parse_sequence(row.get("event_sequence"), [])
            raw_sources = _parse_sequence(row.get("source_sequence"), [])
            raw_gaps = _parse_sequence(row.get("time_gap_sequence"), [])
            original_lengths[row_index] = len(raw_tokens)
            tokens = _truncate_head_tail(raw_tokens, max_events)
            sources = _truncate_head_tail(raw_sources, max_events)
            gaps = _truncate_head_tail(raw_gaps, max_events)
            usable = min(len(tokens), max_events)
            if usable:
                day_tokens[row_index, :usable] = np.asarray(tokens[:usable], dtype=np.uint16)
                day_sources[row_index, :usable] = np.asarray(
                    (sources + [SOURCE_TO_ID["unknown"]] * usable)[:usable],
                    dtype=np.uint8,
                )
                day_gaps[row_index, :usable] = np.asarray(
                    (gaps + [0] * usable)[:usable],
                    dtype=np.float32,
                )

        roles = group.get("role", pd.Series(["UNKNOWN"] * len(group))).fillna("UNKNOWN").astype(str).tolist()
        departments = group.get("department", pd.Series(["UNKNOWN"] * len(group))).fillna("UNKNOWN").astype(str).tolist()
        dates = group["date"].astype(str).tolist()
        labels = pd.to_numeric(group.get("label_day", pd.Series([0] * len(group))), errors="coerce").fillna(0).astype(int).tolist()
        scenarios = group.get("scenario", pd.Series([""] * len(group))).fillna("").astype(str).tolist()
        splits = group.get("split", pd.Series([""] * len(group))).fillna("").astype(str).tolist()

        first_end = 1 if allow_padding else window_size
        for end in range(first_end, len(group) + 1):
            start = max(0, end - window_size)
            window_slices.append((day_offset + start, day_offset + end))
            metas.append({
                "user": user,
                "date": dates[end - 1],
                "role": roles[end - 1],
                "department": departments[end - 1],
                "label_day": labels[end - 1],
                "scenario": scenarios[end - 1],
                "split": splits[end - 1],
                "window_has_positive": bool(any(labels[start:end])),
                "sequence_was_truncated": bool(original_lengths[end - 1] > max_events),
                "original_sequence_length": int(original_lengths[end - 1]),
            })

        count_days.append(count_values)
        token_days.append(day_tokens)
        source_days.append(day_sources)
        gap_days.append(day_gaps)
        day_offset += len(group)

    if not count_days:
        return empty_count, empty_tokens, empty_sources, empty_gaps, [], []
    return (
        np.concatenate(count_days, axis=0),
        np.concatenate(token_days, axis=0),
        np.concatenate(source_days, axis=0),
        np.concatenate(gap_days, axis=0),
        window_slices,
        metas,
    )


def make_multiview_model_inputs(
    df: pd.DataFrame,
    count_columns: list[str],
    window_size: int,
    max_events_per_day: Optional[int] = None,
    allow_padding: bool = True,
):
    """Eager compatibility wrapper for small prediction/calibration inputs."""
    max_events = int(max_events_per_day or FEATURE_RULES["max_events_per_day"])
    count_days, token_days, source_days, gap_days, slices, metas = make_multiview_day_inputs(
        df,
        count_columns,
        window_size,
        max_events_per_day=max_events,
        allow_padding=allow_padding,
    )
    count_windows, token_windows, source_windows, gap_windows = [], [], [], []
    for start, end in slices:
        pad = window_size - (end - start)
        count_block = count_days[start:end]
        token_block = token_days[start:end]
        source_block = source_days[start:end]
        gap_block = gap_days[start:end]
        if pad:
            count_block = np.pad(count_block, ((pad, 0), (0, 0)))
            token_block = np.pad(token_block, ((pad, 0), (0, 0)))
            source_block = np.pad(
                source_block,
                ((pad, 0), (0, 0)),
                constant_values=SOURCE_TO_ID["unknown"],
            )
            gap_block = np.pad(gap_block, ((pad, 0), (0, 0)))
        count_windows.append(count_block)
        token_windows.append(token_block)
        source_windows.append(source_block)
        gap_windows.append(gap_block)
    if not count_windows:
        empty_count = np.empty((0, window_size, len(count_columns)), dtype=np.float32)
        empty_seq = np.empty((0, window_size, max_events), dtype=np.int64)
        return empty_count, empty_seq, empty_seq.copy(), empty_seq.astype(np.float32), []
    return (
        np.stack(count_windows).astype(np.float32),
        np.stack(token_windows).astype(np.int64),
        np.stack(source_windows).astype(np.int64),
        np.stack(gap_windows).astype(np.float32),
        metas,
    )

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
    df = calendarize_multiview(df).sort_values(["user", "date"]).copy()
    for column in feature_cols:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
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
