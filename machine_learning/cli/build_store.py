"""Build/resume the low-RAM disk-backed CERT user-day store."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from insider_ml.contracts import TEST_END
from insider_ml.stream_store import build_stream_store


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Stream raw CERT CSV into compact metadata-only user-day tensors. "
            "Completed raw sources are resumed instead of read again."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=root / "data" / "raw" / "cert4.2")
    parser.add_argument(
        "--store",
        type=Path,
        default=root / "data" / "processed" / "cert4.2_user_days.sqlite",
    )
    parser.add_argument(
        "--max-rows-per-source",
        type=int,
        help="Smoke-only cap; omit for the full raw source",
    )
    parser.add_argument(
        "--end-day",
        type=date.fromisoformat,
        default=TEST_END,
        help="Smoke-only early end is allowed; full experiment uses locked Test end",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.max_rows_per_source is not None and args.max_rows_per_source < 1:
        raise SystemExit("--max-rows-per-source must be positive")
    machine_learning_root = Path(__file__).resolve().parents[1]
    result = build_stream_store(
        raw_root=args.raw_root,
        store_path=args.store,
        feature_config=machine_learning_root / "config" / "feature128.v5.json",
        sequence_config=machine_learning_root / "config" / "sequence7.v4.json",
        max_rows_per_source=args.max_rows_per_source,
        end_day=args.end_day,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
