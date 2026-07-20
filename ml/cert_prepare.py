from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from features import normalize_event_frame


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cert-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--holdout-users", type=int, default=2)
    p.add_argument("--min-logs", type=int, default=50)
    return p.parse_args()


def read_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing {path.name}, skipping.")
        return pd.DataFrame()
    return pd.read_csv(path)


def standardize(cert_dir: Path, file_name: str, event_type: str, default_activity: str | None = None) -> pd.DataFrame:
    df = read_if_exists(cert_dir / file_name)
    if df.empty:
        return df
    df["event_type"] = event_type
    if "activity" not in df.columns and default_activity:
        df["activity"] = default_activity
    return df


def sanitize(value: str) -> str:
    invalid = '<>:"/\\|?*'
    return "".join("_" if ch in invalid else ch for ch in value)


def main():
    args = parse_args()
    cert_dir = Path(args.cert_dir)
    out_dir = Path(args.out_dir)
    holdout_dir = out_dir / "holdout"
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        standardize(cert_dir, "logon.csv", "logon"),
        standardize(cert_dir, "device.csv", "device"),
        standardize(cert_dir, "file.csv", "file", "copy"),
        standardize(cert_dir, "email.csv", "email", "send"),
        standardize(cert_dir, "http.csv", "http", "visit"),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FileNotFoundError(f"No CERT CSV files found in {cert_dir}")

    events = normalize_event_frame(pd.concat(frames, ignore_index=True)).sort_values(["user", "date"])
    counts = events.groupby("user").size().sort_values(ascending=False)
    candidates = counts[counts > args.min_logs]
    if len(candidates) < args.holdout_users:
        raise ValueError(f"Not enough users with > {args.min_logs} logs. Found {len(candidates)}, need {args.holdout_users}")

    holdout_users = list(candidates.head(args.holdout_users).index)
    global_events = events[~events["user"].isin(holdout_users)].copy()
    global_path = out_dir / "global_events.csv"
    global_events.to_csv(global_path, index=False)

    users_index = []
    for user in holdout_users:
        user_events = events[events["user"] == user].copy()
        user_file = holdout_dir / f"{sanitize(user)}.csv"
        user_events.to_csv(user_file, index=False)
        users_index.append({"userId": user, "logCount": int(len(user_events)), "file": str(user_file)})

    (holdout_dir / "index.json").write_text(json.dumps({"users": users_index}, indent=2), encoding="utf-8")
    summary = {"total_events": int(len(events)), "total_users": int(events["user"].nunique()), "global_events": int(len(global_events)), "holdout_users": users_index, "min_logs": args.min_logs, "global_events_path": str(global_path)}
    (out_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
