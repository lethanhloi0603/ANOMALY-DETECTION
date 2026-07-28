"""Locked employee-day evaluation universe derived from effective-dated LDAP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True, slots=True)
class EmployeePeriod:
    employee_id: str
    valid_from: date
    valid_to: date | None


def eligible_employee_days(
    periods: list[EmployeePeriod],
    *,
    start: date,
    end: date,
) -> list[tuple[str, date]]:
    """Return every effective employee-day, including days with zero events."""

    if end < start:
        raise ValueError("end must be on or after start")
    universe: set[tuple[str, date]] = set()
    inclusive_end = end + timedelta(days=1)
    for period in periods:
        employee_id = period.employee_id.strip()
        if not employee_id:
            raise ValueError("employee_id cannot be empty")
        period_end = period.valid_to or inclusive_end
        current = max(start, period.valid_from)
        stop = min(inclusive_end, period_end)
        while current < stop:
            universe.add((employee_id, current))
            current += timedelta(days=1)
    return sorted(universe, key=lambda item: (item[1], item[0]))
