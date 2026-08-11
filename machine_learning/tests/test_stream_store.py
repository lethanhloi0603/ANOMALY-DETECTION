from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np

from insider_ml.dataset import SQLiteWindowDataset
from insider_ml.stream_store import (
    SourceDayAccumulator,
    TemporalReferenceState,
    connect_store,
    decode_day_payload,
    encode_day_payload,
    initialize_store,
    metadata_set,
)


def _encoded_day(*, token: int = 0, second: int = 0) -> bytes:
    feature_values = np.arange(128, dtype=np.float32)
    feature_mask = np.ones(128, dtype=bool)
    tokens = np.zeros(256, dtype=np.uint8)
    pc = np.zeros(256, dtype=np.uint8)
    calendar = np.zeros(256, dtype=np.uint8)
    gaps = np.zeros(256, dtype=np.uint8)
    seconds = np.zeros(256, dtype=np.uint32)
    if token:
        tokens[0] = token
        pc[0] = 1
        calendar[0] = 1
        gaps[0] = 1
        seconds[0] = second
    return encode_day_payload(
        feature_values=feature_values,
        feature_mask=feature_mask,
        tokens=tokens,
        pc_contexts=pc,
        calendar_contexts=calendar,
        gap_buckets=gaps,
        seconds=seconds,
    )


def test_day_payload_round_trip() -> None:
    decoded = decode_day_payload(_encoded_day(token=6, second=3600))
    assert decoded.feature_values.shape == (128,)
    assert decoded.feature_mask.all()
    assert decoded.tokens[0] == 6
    assert decoded.pc_contexts[0] == 1
    assert decoded.seconds[0] == 3600


def test_http_accumulator_never_persists_content_or_raw_url() -> None:
    accumulator = SourceDayAccumulator(
        source="HTTP",
        user_id="U001",
        day=date(2010, 1, 2),
    )
    accumulator.update(
        {
            "id": "E001",
            "pc": "PC-1",
            "url": "https://private.example/secret?token=raw-value",
            "content": "DO-NOT-PERSIST",
        },
        datetime(2010, 1, 2, 8, 0),
    )
    rendered = str(accumulator.payload())
    assert "DO-NOT-PERSIST" not in rendered
    assert "private.example" not in rendered
    assert "raw-value" not in rendered


def test_frozen_temporal_reference_keeps_immutable_train_support() -> None:
    state = TemporalReferenceState()
    role_epoch = "U001|Engineer|2010-01-02"
    start = date(2010, 5, 2)
    for offset in range(30):
        state.add_train_day(
            day=start + timedelta(days=offset),
            user_id="U001",
            role="Engineer",
            role_epoch=role_epoch,
            seconds=[8 * 60 * 60],
        )

    state.freeze()
    frozen = state.select(
        day=date(2010, 6, 1),
        user_id="U001",
        role="Engineer",
        role_epoch=role_epoch,
    )
    assert frozen is not None
    assert frozen.level == "PERSON"
    assert frozen.observation_count == 30

    state.expire(date(2010, 7, 1))
    still_frozen = state.select(
        day=date(2010, 7, 1),
        user_id="U001",
        role="Engineer",
        role_epoch=role_epoch,
    )
    assert still_frozen == frozen


def test_sqlite_window_dataset_reads_thirty_days_without_npz(tmp_path) -> None:
    store = tmp_path / "days.sqlite"
    connection = connect_store(store)
    initialize_store(connection)
    scaler = {
        "schema_version": "robust-feature-scaler.v1",
        "fit_split": "TRAIN",
        "frozen_after_train": True,
        "location": [0.0] * 128,
        "scale": [1.0] * 128,
        "support": [30] * 128,
        "checksum": "a" * 64,
    }
    metadata_set(connection, "daily_materialization_complete", "true")
    metadata_set(connection, "preprocessing_checksum", "b" * 64)
    metadata_set(connection, "scaler_checksum", "a" * 64)
    metadata_set(connection, "scaler", scaler)
    start = date(2010, 1, 2)
    rows = []
    for offset in range(30):
        day = start + timedelta(days=offset)
        rows.append(
            (
                "U001",
                day.isoformat(),
                "TRAIN",
                "Engineer",
                "U001|Engineer|2009-12-01",
                int(offset == 29),
                int(offset == 29),
                128,
                0,
                _encoded_day(
                    token=1 if offset == 29 else 0,
                    second=3600 if offset == 29 else 0,
                ),
            )
        )
    connection.executemany(
        """
        INSERT INTO daily_tensors(
            user_id,day,split,role,role_epoch,event_count,sequence_length,
            feature_observed_count,truncated,payload
        ) VALUES(?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    connection.commit()
    connection.close()

    dataset = SQLiteWindowDataset(store, split="TRAIN")
    batch = dataset[29]
    assert batch["feature_values"].shape == (30, 128)
    assert batch["tokens"].shape == (30, 256)
    assert batch["eligible_day_mask"].all()
    assert batch["token_mask"][-1, 0]
    assert np.isclose(float(batch["time_sin"][-1, 0]), np.sin(np.pi / 12))
    assert np.isclose(float(batch["time_cos"][-1, 0]), np.cos(np.pi / 12))

    weekly = SQLiteWindowDataset(
        store,
        split="TRAIN",
        endpoint_policy="WEEKLY_TRAIN",
    )
    assert 4 <= len(weekly) <= 5
    assert len({date.fromisoformat(item.day).weekday() for item in weekly.samples}) == 1
