"""Materialize LDAP universe and positive-only CERT answer-key user-days."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from insider_ml.cert_data import CERT_TIME_FORMAT, LdapDirectory
from insider_ml.contracts import (
    TEST_END,
    TEST_START,
    VALIDATION_END,
    VALIDATION_START,
)

SPLITS = {
    "VALIDATION": (VALIDATION_START, VALIDATION_END),
    "TEST": (TEST_START, TEST_END),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=root / "data" / "raw" / "cert4.2")
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument("--universe-out", type=Path, required=True)
    parser.add_argument("--labels-out", type=Path, required=True)
    parser.add_argument(
        "--users-from-npz",
        type=Path,
        help="Restrict universe for a smoke/user-shard run; omit for the full LDAP universe",
    )
    return parser.parse_args(argv)


def _answer_rows(raw_root: Path) -> dict[tuple[str, date], dict[str, str]]:
    positive: dict[tuple[str, date], dict[str, str]] = {}
    answer_root = raw_root / "answers"
    for path in sorted(answer_root.glob("r4.2-*/*.csv")):
        scenario = path.parent.name.removeprefix("r4.2-")
        incident_id = path.stem
        subject_user_id = incident_id.rsplit("-", 1)[-1]
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            for line_number, row in enumerate(reader, start=1):
                if len(row) < 5:
                    raise ValueError(f"{path}:{line_number}: malformed answer row")
                timestamp = datetime.strptime(row[2].strip(), CERT_TIME_FORMAT)
                user_id = row[3].strip()
                if user_id != subject_user_id:
                    continue
                key = (user_id, timestamp.date())
                existing = positive.get(key)
                if existing is None:
                    positive[key] = {
                        "user_id": user_id,
                        "day": timestamp.date().isoformat(),
                        "incident_id": incident_id,
                        "scenario": scenario,
                        "is_positive": "true",
                    }
                elif incident_id not in existing["incident_id"].split(";"):
                    existing["incident_id"] += f";{incident_id}"
                    existing["scenario"] += f";{scenario}"
    return positive


def run(args: argparse.Namespace) -> dict[str, object]:
    raw_root = args.raw_root.resolve()
    directory = LdapDirectory.from_raw(raw_root)
    start, end = SPLITS[args.split]
    selected_users: set[str] | None = None
    if args.users_from_npz:
        with np.load(args.users_from_npz, allow_pickle=False) as archive:
            selected_users = {str(value) for value in archive["sample_user_ids"]}

    universe_rows: list[dict[str, str]] = []
    current = start
    while current <= end:
        users = directory.users_on(current)
        if selected_users is not None:
            users &= selected_users
        universe_rows.extend(
            {"user_id": user_id, "day": current.isoformat()}
            for user_id in sorted(users)
        )
        current += timedelta(days=1)
    if not universe_rows:
        raise ValueError(f"LDAP universe is empty for split={args.split}")

    universe_keys = {(row["user_id"], row["day"]) for row in universe_rows}
    labels = [
        row
        for key, row in _answer_rows(raw_root).items()
        if (key[0], key[1].isoformat()) in universe_keys
    ]
    labels.sort(key=lambda row: (row["day"], row["user_id"]))
    args.universe_out.parent.mkdir(parents=True, exist_ok=True)
    with args.universe_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("user_id", "day"))
        writer.writeheader()
        writer.writerows(universe_rows)
    args.labels_out.parent.mkdir(parents=True, exist_ok=True)
    with args.labels_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("user_id", "day", "incident_id", "scenario", "is_positive"),
        )
        writer.writeheader()
        writer.writerows(labels)
    return {
        "schema_version": "cert-evaluation-materialization.v1",
        "split": args.split,
        "universe_user_days": len(universe_rows),
        "positive_user_days": len(labels),
        "positive_definition": (
            "incident subject day containing at least one r4.2 answer-key event; "
            "other actors in the answer file are excluded"
        ),
        "interval_expansion_used": False,
        "users_restricted_from_npz": selected_users is not None,
        "universe_out": str(args.universe_out.resolve()),
        "labels_out": str(args.labels_out.resolve()),
    }


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
