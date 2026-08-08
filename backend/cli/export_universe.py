"""Export the denominator-complete employee-day universe as CSV."""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import engine
from app.domain.universe import EmployeePeriod, eligible_employee_days
from app.models import RoleAssignment, User

SPLIT_RANGES = {
    "TRAIN": (date(2010, 1, 2), date(2010, 5, 31)),
    "VALIDATION": (date(2010, 6, 1), date(2010, 9, 30)),
    "TEST": (date(2010, 10, 1), date(2011, 5, 17)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=tuple(SPLIT_RANGES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start, end = SPLIT_RANGES[args.split]
    with Session(engine) as session:
        rows = session.execute(
            select(
                User.external_user_id,
                RoleAssignment.valid_from,
                RoleAssignment.valid_to,
            ).join(RoleAssignment, RoleAssignment.user_id == User.id)
        )
        periods = [
            EmployeePeriod(
                employee_id=external_user_id,
                valid_from=valid_from,
                valid_to=valid_to,
            )
            for external_user_id, valid_from, valid_to in rows
        ]
    universe = eligible_employee_days(periods, start=start, end=end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("user_id", "day", "split"))
        writer.writerows(
            (employee_id, day.isoformat(), args.split)
            for employee_id, day in universe
        )
    print(f"wrote {len(universe)} employee-days to {args.output}")


if __name__ == "__main__":
    main()
