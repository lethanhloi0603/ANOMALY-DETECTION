"""Prepare one leakage-safe CERT user shard as model-ready NPZ."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from insider_ml.artifacts import atomic_write_json
from insider_ml.cert_data import (
    LdapDirectory,
    build_prepared_days,
    build_windows,
    load_events,
    load_feature_names,
    load_scaler,
    save_scaler,
    save_windows,
    select_users,
)
from insider_ml.contracts import TRAIN_START


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Read CERT r4.2 CSV in place and build Feature128/Sequence7 windows. "
            "The content columns are never retained."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=root / "data" / "raw" / "cert4.2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("TRAIN", "VALIDATION", "TEST"), required=True)
    parser.add_argument("--start-day", type=date.fromisoformat, required=True)
    parser.add_argument("--end-day", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--max-users",
        type=int,
        default=25,
        help="Number of users in this shard, selected deterministically from logon.csv",
    )
    parser.add_argument(
        "--max-rows-per-source",
        type=int,
        help="Smoke-only raw row cap per source; omit for the complete selected date range",
    )
    parser.add_argument("--scaler-in", type=Path)
    parser.add_argument("--scaler-out", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional preparation manifest; defaults beside the NPZ",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.start_day > args.end_day:
        raise ValueError("start-day must be on or before end-day")
    if args.max_users < 1:
        raise ValueError("max-users must be positive")
    if args.max_rows_per_source is not None and args.max_rows_per_source < 1:
        raise ValueError("max-rows-per-source must be positive when supplied")
    if args.split != "TRAIN" and args.scaler_in is None:
        raise ValueError("VALIDATION/TEST preparation requires --scaler-in fitted on Train")

    raw_root = args.raw_root.resolve()
    if not (raw_root / "logon.csv").is_file() or not (raw_root / "LDAP").is_dir():
        raise FileNotFoundError(f"not a CERT r4.2 root: {raw_root}")
    machine_learning_root = Path(__file__).resolve().parents[1]
    feature_config = machine_learning_root / "config" / "feature128.v5.json"
    sequence_config = machine_learning_root / "config" / "sequence7.v4.json"
    feature_names = load_feature_names(feature_config)
    directory = LdapDirectory.from_raw(raw_root)
    users = select_users(
        raw_root,
        start_day=args.start_day,
        end_day=args.end_day,
        max_users=args.max_users,
    )
    history_start = min(args.start_day - timedelta(days=29), TRAIN_START - timedelta(days=29))
    events, rows_read = load_events(
        raw_root,
        users,
        start_day=TRAIN_START,
        end_day=args.end_day,
        max_rows_per_source=args.max_rows_per_source,
    )
    prepared, pc_profiles = build_prepared_days(
        events=events,
        users=users,
        directory=directory,
        feature_names=feature_names,
        start_day=history_start,
        end_day=args.end_day,
    )
    scaler = load_scaler(args.scaler_in) if args.scaler_in else None
    windows, fitted_scaler = build_windows(
        prepared_days=prepared,
        directory=directory,
        users=users,
        start_day=args.start_day,
        end_day=args.end_day,
        split=args.split,
        feature_config=feature_config,
        sequence_config=sequence_config,
        pc_profiles=pc_profiles,
        scaler=scaler,
    )
    save_windows(args.output, windows)
    if args.scaler_out:
        save_scaler(args.scaler_out, fitted_scaler)
    manifest = {
        "schema_version": "cert-preparation-manifest.v1",
        "raw_root": str(raw_root),
        "raw_data_copied": False,
        "content_policy": "METADATA_ONLY",
        "timestamp_basis": "CERT_LOCAL_WALL_CLOCK",
        "split": args.split,
        "start_day": args.start_day.isoformat(),
        "end_day": args.end_day.isoformat(),
        "history_window": f"[{history_start.isoformat()},{args.end_day.isoformat()}]",
        "users": list(users),
        "selected_user_count": len(users),
        "max_users": args.max_users,
        "events_retained": len(events),
        "raw_rows_read": rows_read,
        "samples": int(windows.feature_values.shape[0]),
        "active_sequence_days": int(windows.token_mask.any(axis=2).sum()),
        "truncated_sequence_days": sum(
            count > 256
            for count in Counter((event.user_id, event.day) for event in events).values()
        ),
        "feature_schema_version": str(windows.feature_schema_version.item()),
        "sequence_schema_version": str(windows.sequence_schema_version.item()),
        "preprocessing_checksum": str(windows.preprocessing_checksum.item()),
        "scaler_checksum": str(windows.scaler_checksum.item()),
        "endpoint_policy": "ALL",
        "smoke_row_cap_per_source": args.max_rows_per_source,
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
    }
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    atomic_write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
