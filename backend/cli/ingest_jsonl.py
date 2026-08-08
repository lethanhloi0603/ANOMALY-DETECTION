"""Stream canonical JSONL events to the backend with source/month checkpoints.

The script never loads the full input into memory. It performs one streaming
pass to compute the checksum and row count, then a second pass to submit bounded
batches. Input must be sorted by timestamp so a source/month checkpoint is
monotonic and resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


def file_manifest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                rows += 1
    return digest.hexdigest(), rows


class ApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None,
        actor: str,
        timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.actor = actor
        self.timeout = timeout

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = (
            json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        )
        headers = {
            "accept": "application/json",
            "x-actor": self.actor,
        }
        if body is not None:
            headers["content-type"] = "application/json"
        if self.api_key:
            headers["x-api-key"] = self.api_key
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content = response.read()
                return json.loads(content) if content else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"API {method} {path} failed with HTTP {exc.code}: {detail}"
            ) from exc


def event_month(event: dict[str, Any]) -> tuple[datetime, str]:
    raw_timestamp = event.get("timestamp")
    if not isinstance(raw_timestamp, str):
        raise ValueError("every event must contain an ISO timestamp string")
    timestamp = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("event timestamps must include a timezone offset")
    return timestamp, f"{timestamp.year:04d}-{timestamp.month:02d}"


def existing_offsets(client: ApiClient, job_id: str) -> dict[str, int]:
    checkpoints = client.request("GET", f"/api/v1/ingestions/{job_id}/checkpoints")
    return {
        checkpoint["partition_key"]: int(checkpoint["row_offset"]) for checkpoint in checkpoints
    }


def ingest(
    path: Path,
    *,
    source: str,
    client: ApiClient,
    batch_size: int,
    progress_every: int,
) -> None:
    checksum, total_rows = file_manifest(path)
    job = client.request(
        "POST",
        "/api/v1/ingestions",
        {
            "source": source,
            "input_uri": str(path.resolve()),
            "original_filename": path.name,
            "sha256": checksum,
            "idempotency_key": f"{source.lower()}:{checksum}",
            "schema_version": "canonical-event.v1",
            "total_rows": total_rows,
        },
    )
    job_id = job["id"]
    offsets = existing_offsets(client, job_id)
    started = time.monotonic()
    batch: list[dict[str, Any]] = []
    batch_month: str | None = None
    partition_seen = 0
    processed_input = 0
    previous_timestamp: datetime | None = None

    def flush(month: str, seen: int) -> None:
        nonlocal batch
        if not batch:
            return
        result = client.request(
            "POST",
            "/api/v1/events/batch",
            {"ingestion_job_id": job_id, "events": batch},
        )
        last = batch[-1]
        client.request(
            "PUT",
            f"/api/v1/ingestions/{job_id}/checkpoints",
            {
                "partition_key": f"{source}/{month}",
                "row_offset": seen,
                "cursor": {
                    "event_uid": last["event_uid"],
                    "timestamp": last["timestamp"],
                },
                "input_checksum": checksum,
            },
        )
        print(
            f"partition={source}/{month} offset={seen} "
            f"inserted={result['inserted']} duplicates={result['duplicates']}",
            flush=True,
        )
        batch = []

    with path.open(encoding="utf-8") as handle:
        for file_line, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            event = json.loads(raw_line)
            timestamp, month = event_month(event)
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(f"input is not timestamp-sorted at physical line {file_line}")
            previous_timestamp = timestamp
            if str(event.get("source", "")).upper() != source:
                raise ValueError(
                    f"event source mismatch at physical line {file_line}: expected {source}"
                )
            if batch_month is None:
                batch_month = month
                partition_seen = 0
            elif month != batch_month:
                flush(batch_month, partition_seen)
                batch_month = month
                partition_seen = 0

            partition_seen += 1
            processed_input += 1
            checkpoint_key = f"{source}/{month}"
            if partition_seen <= offsets.get(checkpoint_key, 0):
                continue
            batch.append(event)
            if len(batch) >= batch_size:
                flush(month, partition_seen)

            if processed_input % progress_every == 0:
                elapsed = max(time.monotonic() - started, 0.001)
                rate = processed_input / elapsed
                remaining = max(total_rows - processed_input, 0)
                eta = remaining / rate if rate else 0
                print(
                    f"progress={processed_input}/{total_rows} "
                    f"rate={rate:.1f} rows/s eta={eta:.1f}s",
                    flush=True,
                )
    if batch_month is not None:
        flush(batch_month, partition_seen)
    elapsed = time.monotonic() - started
    print(
        f"complete job_id={job_id} rows={processed_input} elapsed={elapsed:.1f}s",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream canonical JSONL events to the Insider Threat API"
    )
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--source",
        required=True,
        choices=["LOGON", "DEVICE", "FILE", "HTTP", "EMAIL"],
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key")
    parser.add_argument("--actor", default="canonical-jsonl-ingestor")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 1000:
        parser.error("--batch-size must be within [1, 1000]")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if not args.path.is_file():
        parser.error(f"input file does not exist: {args.path}")
    return args


def main() -> None:
    args = parse_args()
    client = ApiClient(
        args.api_url,
        api_key=args.api_key,
        actor=args.actor,
        timeout=args.timeout,
    )
    try:
        ingest(
            args.path,
            source=args.source,
            client=client,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ingestion failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
