# Backend

FastAPI, persistence, Person → Role → Global readiness, score fusion, alerts,
audit, and safe-update workflow.

Run commands from this directory:

```powershell
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
..\.venv\Scripts\python.exe -m pytest
..\.venv\Scripts\python.exe -m alembic upgrade head
..\.venv\Scripts\python.exe -m cli.init_db
```

Runtime data is stored outside this package at `../data/runtime`; ML contracts
are read from `../machine_learning/config`.

Scoring requires `LOCKED_ALERT_THRESHOLD`, exported from the label-free
Validation 99.5th-percentile decision and frozen before Test. Primary
Validation/Test references are Train-fitted and immutable; safe personalized
updates run only for `PRODUCTION` after the closed quarantine window `[D,D+30]`;
the earliest eligible day is `D+31` and every day in the window requires a
complete scoring watermark.

Personal reference materialization is fail-closed behind both the versioned
framework gate and `SAFE_UPDATE_MATERIALIZATION_ENABLED`. Keep both disabled for
shadow rollout. See [`../docs/safe-update-rollout.md`](../docs/safe-update-rollout.md)
for the activation prerequisites.

The scoring request supplies raw branch errors only. Readiness support and
calibration come from stored pipeline-built references. Sequence PC context,
event gaps, calendar context, and cyclic time are derived from canonical events;
clients cannot declare them. Evaluation denominators are exported from
effective-dated LDAP assignments, including employee-days with zero events:

```powershell
..\.venv\Scripts\python.exe -m cli.export_universe `
  --split TEST `
  --output ..\data\artifacts\reports\test_universe.csv
```
