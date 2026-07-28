from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.catalog import load_feature_catalog
from app.database import Base, get_db, init_db, make_engine
from app.main import create_app
from app.models import AuditLog, FeatureCatalog, User, UserDayFeature
from app.settings import settings


@contextmanager
def configured_client(**setting_overrides: Any) -> Iterator[TestClient]:
    engine = make_engine("sqlite+pysqlite:///:memory:")
    init_db(engine)
    session_factory = sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )
    overrides = {
        "database_url": "sqlite+pysqlite:///:memory:",
        "app_env": "development",
        "auto_create_schema": False,
        "api_key": None,
        "api_key_actor": "test-api-principal",
        "scorer_api_key": None,
        "scorer_api_key_actor": "test-scorer-principal",
        "locked_alert_threshold": 0.95,
        "cors_origins": (),
    }
    overrides.update(setting_overrides)
    test_settings = replace(settings, **overrides)
    app = create_app(test_settings, database_engine=engine)
    app.state.test_engine = engine
    app.state.test_session_factory = session_factory

    def override_db():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as test_client:
        yield test_client
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def client():
    with configured_client() as test_client:
        yield test_client


def create_identity(
    client: TestClient,
    *,
    headers: dict[str, str] | None = None,
) -> None:
    response = client.post(
        "/api/v1/users",
        json={"user_id": "U001", "display_name": "Alice"},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    response = client.post(
        "/api/v1/roles",
        json={"role_code": "ENGINEER", "display_name": "Engineer"},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    response = client.post(
        "/api/v1/users/U001/role-assignments",
        json={
            "role_code": "ENGINEER",
            "valid_from": "2010-01-02",
            "source_snapshot_date": "2010-01-01",
        },
        headers=headers,
    )
    assert response.status_code == 201, response.text


def test_health_framework_and_openapi(client: TestClient) -> None:
    assert client.get("/health/live").json() == {"status": "alive"}
    assert client.get("/health/ready").json() == {"status": "ready"}
    framework = client.get("/api/v1/framework")
    assert framework.status_code == 200
    assert framework.json()["feature_schema_version"] == "feature128.v5"
    assert framework.json()["feature_dimension"] == 128
    assert framework.json()["sequence_schema_version"] == "sequence7.v4"
    assert len(framework.json()["sequence_tokens"]) == 7
    assert framework.json()["framework_config"]["temporal_baseline"][
        "fixed_business_hours"
    ] is None
    config = framework.json()["framework_config"]
    assert config["primary_evaluation"]["all_reference_levels_train_only"] is True
    assert config["primary_evaluation"]["client_supplied_support_allowed"] is False
    assert config["safe_personalized_update"]["primary_evaluation_enabled"] is False
    assert config["evaluation_universe"]["include_inactive_days"] is True
    assert config["timestamp_policy"]["cert_basis"] == "LOCAL_WALL_CLOCK"
    assert config["window_policy"]["range"] == "[D-29,D]"
    feature_names = {item["name"] for item in load_feature_catalog()["features"]}
    assert not any("after_hours" in name or "business_hours" in name for name in feature_names)
    assert "unusual_time_relative_to_baseline_mean" in feature_names
    openapi = client.get("/openapi.json").json()
    assert "/api/v1/assessments/score" in openapi["paths"]
    assert "/api/v1/events/batch" in openapi["paths"]


def test_production_requires_general_and_scorer_api_keys(client: TestClient) -> None:
    engine = client.app.state.test_engine
    with pytest.raises(RuntimeError, match="API_KEY"):
        create_app(
            replace(
                settings,
                app_env="production",
                api_key=None,
                scorer_api_key="scorer-secret",
                locked_alert_threshold=0.95,
            ),
            database_engine=engine,
        )
    with pytest.raises(RuntimeError, match="SCORER_API_KEY"):
        create_app(
            replace(
                settings,
                app_env="production",
                api_key="general-secret",
                scorer_api_key=None,
                locked_alert_threshold=0.95,
            ),
            database_engine=engine,
        )
    with pytest.raises(RuntimeError, match="must be distinct"):
        create_app(
            replace(
                settings,
                app_env="production",
                api_key="shared-secret",
                scorer_api_key="shared-secret",
                locked_alert_threshold=0.95,
            ),
            database_engine=engine,
        )

    with pytest.raises(RuntimeError, match="LOCKED_ALERT_THRESHOLD"):
        create_app(
            replace(
                settings,
                app_env="production",
                api_key="general-secret",
                scorer_api_key="scorer-secret",
                locked_alert_threshold=None,
            ),
            database_engine=engine,
        )


def test_authenticated_principals_ignore_spoofed_actor_header() -> None:
    with configured_client(
        api_key="general-secret",
        api_key_actor="trusted-api-client",
        scorer_api_key="scorer-secret",
        scorer_api_key_actor="trusted-scoring-worker",
    ) as secured:
        general_headers = {
            "x-api-key": "general-secret",
            "x-actor": "spoofed-client",
        }
        assert secured.get("/api/v1/users/U001").status_code == 401
        create_identity(secured, headers=general_headers)

        session_factory = secured.app.state.test_session_factory
        with session_factory() as session:
            actors = set(session.scalars(select(AuditLog.actor)).all())
        assert actors == {"trusted-api-client"}

        payload = {
            "user_id": "U001",
            "day": "2010-06-01",
            "model_version": "auth.v1",
            "config_version": "framework.v4",
            "split": "VALIDATION",
        }
        denied = secured.post(
            "/api/v1/assessments/score",
            json=payload,
            headers=general_headers,
        )
        assert denied.status_code == 401
        assert denied.json()["code"] == "SCORER_UNAUTHORIZED"

        scored = secured.post(
            "/api/v1/assessments/score",
            json=payload,
            headers={
                **general_headers,
                "x-scorer-api-key": "scorer-secret",
            },
        )
        assert scored.status_code == 201, scored.text
        with session_factory() as session:
            scoring_actor = session.scalar(
                select(AuditLog.actor).where(AuditLog.action == "risk_assessment.created")
            )
        assert scoring_actor == "trusted-scoring-worker"


def test_readiness_requires_schema_and_active_128_feature_catalog(
    client: TestClient,
) -> None:
    session_factory = client.app.state.test_session_factory
    with session_factory() as session:
        catalog = session.scalar(select(FeatureCatalog))
        assert catalog is not None
        catalog.is_active = False
        session.commit()

    unavailable = client.get("/health/ready")
    assert unavailable.status_code == 503
    assert unavailable.json()["code"] == "FEATURE_CATALOG_NOT_READY"

    with session_factory() as session:
        catalog = session.scalar(select(FeatureCatalog))
        assert catalog is not None
        catalog.is_active = True
        session.commit()

    Base.metadata.tables["alerts"].drop(client.app.state.test_engine)
    unavailable = client.get("/health/ready")
    assert unavailable.status_code == 503
    assert unavailable.json()["code"] == "DATABASE_SCHEMA_NOT_READY"
    assert "alerts" in unavailable.json()["details"]["missing_tables"]


def test_invalid_alert_status_query_returns_422(client: TestClient) -> None:
    response = client.get("/api/v1/alerts", params={"status": "not-a-status"})
    assert response.status_code == 422
    assert response.json()["code"] == "REQUEST_VALIDATION_ERROR"


def test_request_body_limit_applies_without_content_length() -> None:
    with configured_client(max_request_bytes=64) as limited:
        chunks = iter((b'{"events":[', b"x" * 80, b"]}"))
        response = limited.post(
            "/api/v1/events/batch",
            content=chunks,
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 413, response.text
    assert response.json()["code"] == "REQUEST_TOO_LARGE"


def test_event_ingestion_is_idempotent_and_blocks_labels(client: TestClient) -> None:
    create_identity(client)
    event = {
        "event_uid": "file:1",
        "source": "FILE",
        "original_id": "1",
        "timestamp": "2010-06-01T09:00:00+00:00",
        "user_id": "U001",
        "pc": "PC-1",
        "action": "FILE_COPY",
        "object": "report.pdf",
        "source_payload": {"filename": "report.pdf"},
    }
    first = client.post("/api/v1/events/batch", json={"events": [event]})
    assert first.status_code == 202, first.text
    assert first.json()["inserted"] == 1

    second = client.post("/api/v1/events/batch", json={"events": [event]})
    assert second.status_code == 202, second.text
    assert second.json()["inserted"] == 0
    assert second.json()["duplicates"] == 1

    poisoned = {
        **event,
        "event_uid": "file:2",
        "original_id": "2",
        "source_payload": {"nested": {"scenario": "answer-key"}},
    }
    rejected = client.post("/api/v1/events/batch", json={"events": [poisoned]})
    assert rejected.status_code == 422
    assert rejected.json()["code"] == "LABEL_FIREWALL_VIOLATION"

    unsupported_http = {
        "event_uid": "http:1",
        "source": "HTTP",
        "original_id": "http-1",
        "timestamp": "2010-06-01T09:01:00+00:00",
        "user_id": "U001",
        "pc": "PC-1",
        "action": "BROWSE",
        "object": "https://example.test/",
        "source_payload": {"url": "https://example.test/", "method": "GET"},
    }
    rejected = client.post("/api/v1/events/batch", json={"events": [unsupported_http]})
    assert rejected.status_code == 422
    assert rejected.json()["code"] == "CERT_HTTP_FIELD_UNSUPPORTED"


def test_feature_and_sequence_storage_contracts(client: TestClient) -> None:
    create_identity(client)
    names = [item["name"] for item in load_feature_catalog()["features"]]
    values = {name: 0.0 for name in names}
    values["logon_count"] = 2
    values["total_event_count"] = 2
    feature = client.put(
        "/api/v1/user-days/U001/2010-06-01/features",
        json={"values": values, "context": {"source": "fixture"}},
    )
    assert feature.status_code == 200, feature.text
    assert len(feature.json()["values_json"]) == 128
    content_feature = client.put(
        "/api/v1/user-days/U001/2010-06-02/features",
        json={"values": values, "context": {"content_keywords": ["secret"]}},
    )
    assert content_feature.status_code == 422
    assert content_feature.json()["code"] == "PRIMARY_METADATA_ONLY_VIOLATION"

    size = 300
    start = datetime(2010, 6, 1, 9, 0, tzinfo=UTC)
    events = [
        {
            "event_uid": f"event-{index:03d}",
            "source": "LOGON",
            "original_id": f"logon-{index:03d}",
            "timestamp": (start + timedelta(seconds=index)).isoformat(),
            "user_id": "U001",
            "pc": "PC-1",
            "action": "LOGON",
            "source_payload": {},
        }
        for index in range(size)
    ]
    ingested = client.post("/api/v1/events/batch", json={"events": events})
    assert ingested.status_code == 202, ingested.text
    assert ingested.json()["inserted"] == size

    sequence = client.put(
        "/api/v1/user-days/U001/2010-06-01/sequence",
        json={
            "tokens": ["LOGON"] * size,
            "event_uids": [f"event-{index:03d}" for index in range(size)],
            "side_fields": [{} for _ in range(size)],
        },
    )
    assert sequence.status_code == 200, sequence.text
    body = sequence.json()
    assert body["seq_len"] == 300
    assert body["truncated"] is True
    assert len(body["tokens_json"]) == 256
    assert body["event_uids_json"][127] == "event-127"
    assert body["event_uids_json"][128] == "event-172"
    assert set(body["pc_contexts_json"]) == {"UNKNOWN"}
    assert body["gap_buckets_json"][0] is None
    assert set(body["gap_buckets_json"][1:]) == {"0-1"}
    assert set(body["calendar_contexts_json"]) == {"WEEKDAY"}
    assert len(body["time_sin_json"]) == 256
    assert len(body["time_cos_json"]) == 256
    assert all(
        sin_value**2 + cos_value**2 == pytest.approx(1.0)
        for sin_value, cos_value in zip(
            body["time_sin_json"],
            body["time_cos_json"],
            strict=True,
        )
    )

    reserved_time_field = client.put(
        "/api/v1/user-days/U001/2010-06-01/sequence",
        json={
            "tokens": ["LOGON"],
            "event_uids": ["event-000"],
            "side_fields": [{"time_sin": 0.0}],
        },
    )
    assert reserved_time_field.status_code == 422

    content_side_field = client.put(
        "/api/v1/user-days/U001/2010-06-01/sequence",
        json={
            "tokens": ["LOGON"],
            "event_uids": ["event-000"],
            "side_fields": [{"message_body": "secret"}],
        },
    )
    assert content_side_field.status_code == 422
    assert content_side_field.json()["code"] == "PRIMARY_METADATA_ONLY_VIOLATION"

    self_declared_context = client.put(
        "/api/v1/user-days/U001/2010-06-01/sequence",
        json={
            "tokens": ["LOGON"],
            "event_uids": ["event-000"],
            "pc_contexts": ["OWN"],
            "gap_buckets": [None],
        },
    )
    assert self_declared_context.status_code == 422
    assert self_declared_context.json()["code"] == "REQUEST_VALIDATION_ERROR"


def test_cert_timestamp_preserves_local_wall_clock_without_timezone_claim(
    client: TestClient,
) -> None:
    create_identity(client)
    event = {
        "event_uid": "wall-clock:1",
        "source": "LOGON",
        "original_id": "wall-clock:1",
        "timestamp": "2010-06-01T00:30:00+07:00",
        "user_id": "U001",
        "action": "LOGON",
        "source_payload": {},
    }
    ingested = client.post("/api/v1/events/batch", json={"events": [event]})
    assert ingested.status_code == 202, ingested.text
    stored = client.get(
        "/api/v1/events",
        params={"user_id": "U001", "day": "2010-06-01"},
    )
    assert stored.status_code == 200, stored.text
    assert stored.json()[0]["event_date"] == "2010-06-01"

    sequence = client.put(
        "/api/v1/user-days/U001/2010-06-01/sequence",
        json={"tokens": ["LOGON"], "event_uids": ["wall-clock:1"]},
    )
    assert sequence.status_code == 200, sequence.text
    angle = 2 * math.pi * 30 / 1440
    assert sequence.json()["time_sin_json"][0] == pytest.approx(math.sin(angle))
    assert sequence.json()["time_cos_json"][0] == pytest.approx(math.cos(angle))


def test_sequence_pc_context_is_fitted_from_train_only_user_days(
    client: TestClient,
) -> None:
    create_identity(client)
    for index in range(2, 6):
        created = client.post(
            "/api/v1/users",
            json={"user_id": f"U{index:03d}", "display_name": f"User {index}"},
        )
        assert created.status_code == 201, created.text
    train_events = [
        {
            "event_uid": f"shared-train:{index}",
            "source": "FILE",
            "original_id": f"shared-train:{index}",
            "timestamp": "2010-01-02T09:00:00",
            "user_id": f"U{index:03d}",
            "pc": "PC-SHARED",
            "action": "FILE_COPY",
            "source_payload": {},
        }
        for index in range(1, 6)
    ]
    validation_event = {
        "event_uid": "shared-validation:1",
        "source": "FILE",
        "original_id": "shared-validation:1",
        "timestamp": "2010-06-01T09:00:00",
        "user_id": "U001",
        "pc": "PC-SHARED",
        "action": "FILE_COPY",
        "source_payload": {},
    }
    ingested = client.post(
        "/api/v1/events/batch",
        json={"events": [*train_events, validation_event]},
    )
    assert ingested.status_code == 202, ingested.text
    sequence = client.put(
        "/api/v1/user-days/U001/2010-06-01/sequence",
        json={"tokens": ["FILE"], "event_uids": ["shared-validation:1"]},
    )
    assert sequence.status_code == 200, sequence.text
    assert sequence.json()["pc_contexts_json"] == ["SHARED"]


def _stored_active_day(client: TestClient, user_id: str, day: date) -> bool:
    with Session(client.app.state.test_engine) as session:
        value = session.scalar(
            select(UserDayFeature.is_active_day)
            .join(User, User.id == UserDayFeature.user_id)
            .where(
                User.external_user_id == user_id,
                UserDayFeature.day == day,
            )
        )
    assert value is not None
    return bool(value)


def test_active_day_uses_canonical_events_or_total_event_count_only(
    client: TestClient,
) -> None:
    create_identity(client)
    names = [item["name"] for item in load_feature_catalog()["features"]]
    values = {name: 0.0 for name in names}
    values["active_days_last_30"] = 12

    response = client.put(
        "/api/v1/user-days/U001/2010-06-01/features",
        json={"values": values},
    )
    assert response.status_code == 200, response.text
    assert _stored_active_day(client, "U001", date(2010, 6, 1)) is False

    event = {
        "event_uid": "active-day-event",
        "source": "FILE",
        "original_id": "active-day-event",
        "timestamp": "2010-06-01T09:00:00+00:00",
        "user_id": "U001",
        "action": "FILE_COPY",
        "source_payload": {},
    }
    ingested = client.post("/api/v1/events/batch", json={"events": [event]})
    assert ingested.status_code == 202, ingested.text
    response = client.put(
        "/api/v1/user-days/U001/2010-06-01/features",
        json={"values": values},
    )
    assert response.status_code == 200, response.text
    assert _stored_active_day(client, "U001", date(2010, 6, 1)) is True

    explicit_total = dict(values)
    explicit_total["active_days_last_30"] = 0
    explicit_total["total_event_count"] = 1
    response = client.put(
        "/api/v1/user-days/U001/2010-06-02/features",
        json={"values": explicit_total},
    )
    assert response.status_code == 200, response.text
    assert _stored_active_day(client, "U001", date(2010, 6, 2)) is True


def _post_canonical_event(
    client: TestClient,
    *,
    event_uid: str,
    source: str,
    action: str,
    timestamp: str,
    user_id: str = "U001",
) -> None:
    response = client.post(
        "/api/v1/events/batch",
        json={
            "events": [
                {
                    "event_uid": event_uid,
                    "source": source,
                    "original_id": event_uid,
                    "timestamp": timestamp,
                    "user_id": user_id,
                    "action": action,
                    "source_payload": {},
                }
            ]
        },
    )
    assert response.status_code == 202, response.text


def _sequence_payload(
    event_uids: list[str],
    tokens: list[str],
) -> dict[str, object]:
    return {
        "tokens": tokens,
        "event_uids": event_uids,
        "side_fields": [{} for _ in event_uids],
    }


def test_sequence_requires_canonical_context_order_and_token_mapping(
    client: TestClient,
) -> None:
    create_identity(client)
    timestamp = "2010-06-01T09:00:00+00:00"
    event_specs = [
        ("01-logon", "LOGON", "LOGON", "LOGON"),
        ("02-logoff", "LOGON", "LOGOFF", "LOGOFF"),
        ("03-device-connect", "DEVICE", "CONNECT", "DEVICE_CONNECT"),
        ("04-device-disconnect", "DEVICE", "DISCONNECT", "DEVICE_DISCONNECT"),
        ("05-file", "FILE", "FILE_COPY", "FILE"),
        ("06-http", "HTTP", "REQUEST", "HTTP"),
        ("07-email", "EMAIL", "SEND", "EMAIL"),
    ]
    for event_uid, source, action, _token in event_specs:
        _post_canonical_event(
            client,
            event_uid=event_uid,
            source=source,
            action=action,
            timestamp=timestamp,
        )

    event_uids = [item[0] for item in event_specs]
    tokens = [item[3] for item in event_specs]
    accepted = client.put(
        "/api/v1/user-days/U001/2010-06-01/sequence",
        json=_sequence_payload(event_uids, tokens),
    )
    assert accepted.status_code == 200, accepted.text

    missing = client.put(
        "/api/v1/user-days/U001/2010-06-01/sequence",
        json=_sequence_payload(["does-not-exist"], ["FILE"]),
    )
    assert missing.status_code == 422
    assert missing.json()["code"] == "SEQUENCE_EVENT_UID_NOT_FOUND"

    reversed_order = client.put(
        "/api/v1/user-days/U001/2010-06-01/sequence",
        json=_sequence_payload(
            [event_uids[1], event_uids[0], *event_uids[2:]],
            [tokens[1], tokens[0], *tokens[2:]],
        ),
    )
    assert reversed_order.status_code == 422
    assert reversed_order.json()["code"] == "SEQUENCE_EVENT_ORDER_INVALID"

    wrong_token = list(tokens)
    wrong_token[3] = "DEVICE_CONNECT"
    token_mismatch = client.put(
        "/api/v1/user-days/U001/2010-06-01/sequence",
        json=_sequence_payload(event_uids, wrong_token),
    )
    assert token_mismatch.status_code == 422
    assert token_mismatch.json()["code"] == "SEQUENCE_TOKEN_SOURCE_MISMATCH"

    second_user = client.post(
        "/api/v1/users",
        json={"user_id": "U002", "display_name": "Bob"},
    )
    assert second_user.status_code == 201, second_user.text
    _post_canonical_event(
        client,
        event_uid="other-user-event",
        source="FILE",
        action="FILE_COPY",
        timestamp=timestamp,
        user_id="U002",
    )
    wrong_context = client.put(
        "/api/v1/user-days/U001/2010-06-01/sequence",
        json=_sequence_payload(["other-user-event"], ["FILE"]),
    )
    assert wrong_context.status_code == 422
    assert wrong_context.json()["code"] == "SEQUENCE_EVENT_CONTEXT_MISMATCH"


def create_reference(
    client: TestClient,
    *,
    branch: str,
    level: str,
    scope_key: str,
    checksum_char: str,
    support: dict,
    frozen: bool = True,
    as_of_date: str = "2010-06-01",
    fitted_through: str = "2010-05-31",
) -> str:
    response = client.post(
        "/api/v1/references",
        json={
            "branch": branch,
            "level": level,
            "scope_key": scope_key,
            "as_of_date": as_of_date,
            "model_version": "baseline.v1",
            "config_version": "framework.v4",
            "version": f"{branch.lower()}-{level.lower()}-{checksum_char}.v1",
            "fitted_through": fitted_through,
            "frozen": frozen,
            "support": {
                **(
                    {
                        "active_days": 40,
                        "span_days": 60,
                        "active_days_current_role": 40,
                        "coverage": 1.0,
                        "min_feature_observations": 30,
                        "last_active_gap_days": 1,
                    }
                    if branch == "FEATURE" and level == "PERSON"
                    else {}
                ),
                **(
                    {
                        "sequence_days": 25,
                        "transitions": 600,
                        "span_days": 40,
                        "sequence_days_current_role": 25,
                        "last_active_gap_days": 1,
                    }
                    if branch == "SEQUENCE" and level == "PERSON"
                    else {}
                ),
                **(
                    {
                        "peer_users": 20,
                        "sequence_days": 400,
                        "transitions": 20_000,
                        "recent_transitions": 3_000,
                    }
                    if branch == "SEQUENCE" and level == "ROLE"
                    else {}
                ),
                **support,
            },
            "statistics": {
                "calibrator": {
                    "method": "empirical_cdf.v1",
                    "sorted_scores": [0.0, 0.1, 0.2, 0.3, 0.4, 1.0],
                }
            },
            "checksum": checksum_char * 64,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_all_evaluation_reference_levels_are_frozen_train_only(
    client: TestClient,
) -> None:
    create_identity(client)
    mutable_person = create_reference(
        client,
        branch="FEATURE",
        level="PERSON",
        scope_key="user:U001",
        checksum_char="d",
        support={},
        frozen=False,
    )
    rejected_mutable = client.post(
        "/api/v1/assessments/score",
        json={
            "user_id": "U001",
            "day": "2010-06-01",
            "model_version": "baseline.v1",
            "config_version": "framework.v4",
            "split": "VALIDATION",
            "feature": {
                "raw_score": 1.0,
                "reference_profile_ids": {"PERSON": mutable_person},
            },
        },
    )
    assert rejected_mutable.status_code == 422, rejected_mutable.text
    assert rejected_mutable.json()["code"] == "REFERENCE_NOT_FROZEN"

    post_train_person = create_reference(
        client,
        branch="FEATURE",
        level="PERSON",
        scope_key="user:U001",
        checksum_char="e",
        support={},
        as_of_date="2010-06-02",
        fitted_through="2010-06-01",
    )
    rejected_post_train = client.post(
        "/api/v1/assessments/score",
        json={
            "user_id": "U001",
            "day": "2010-06-02",
            "model_version": "baseline.v1",
            "config_version": "framework.v4",
            "split": "VALIDATION",
            "feature": {
                "raw_score": 1.0,
                "reference_profile_ids": {"PERSON": post_train_person},
            },
        },
    )
    assert rejected_post_train.status_code == 422, rejected_post_train.text
    assert rejected_post_train.json()["code"] == "REFERENCE_NOT_TRAIN_ONLY"


def test_independent_backoff_fusion_alert_and_frozen_evaluation_update(
    client: TestClient,
) -> None:
    create_identity(client)
    feature_person = create_reference(
        client,
        branch="FEATURE",
        level="PERSON",
        scope_key="user:U001",
        checksum_char="a",
        support={"active_days": 40, "coverage": 1.0},
    )
    sequence_person = create_reference(
        client,
        branch="SEQUENCE",
        level="PERSON",
        scope_key="user:U001",
        checksum_char="b",
        support={"sequence_days": 25, "transitions": 400},
    )
    sequence_role = create_reference(
        client,
        branch="SEQUENCE",
        level="ROLE",
        scope_key="role:ENGINEER",
        checksum_char="c",
        support={"sequence_days": 400, "transitions": 20_000},
    )

    response = client.post(
        "/api/v1/assessments/score",
        json={
            "user_id": "U001",
            "day": "2010-06-01",
            "model_version": "baseline.v1",
            "config_version": "framework.v4",
            "split": "VALIDATION",
            "feature": {
                "raw_score": 2.4,
                "reference_profile_ids": {"PERSON": feature_person},
                "evidence": {"top_features": ["file_copy_count"]},
            },
            "sequence": {
                "raw_score": 1.8,
                "reference_profile_ids": {
                    "PERSON": sequence_person,
                    "ROLE": sequence_role,
                },
                "seq_len": 20,
                "truncated": False,
                "evidence": {"top_transitions": ["FILE->EMAIL"]},
            },
            "feature_weight": 0.5,
            "sequence_weight": 0.5,
            "alert_threshold": 0.95,
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["risk"] == pytest.approx(1.0)
    assert response.json()["is_alert"] is True

    detail = client.get(
        "/api/v1/assessments/U001/2010-06-01",
        params={
            "model_version": "baseline.v1",
            "config_version": "framework.v4",
        },
    )
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["feature"]["selected_level"] == "PERSON"
    assert body["sequence"]["selected_level"] == "ROLE"
    assert "SP_TRANS_LT_500" in body["sequence"]["fallback_reasons_json"]
    assert body["alert"]["status"] == "OPEN"
    assert body["safe_updates"] == []

    processed = client.post(
        "/api/v1/safe-updates/process",
        json={"through_date": "2010-06-08"},
    )
    assert processed.status_code == 200, processed.text
    assert processed.json() == {"accepted": 0, "rejected": 0, "pending": 0}
    detail = client.get(
        "/api/v1/assessments/U001/2010-06-01",
        params={
            "model_version": "baseline.v1",
            "config_version": "framework.v4",
        },
    )
    assert detail.json()["safe_updates"] == []


def test_assessment_rejects_client_supplied_readiness_support(client: TestClient) -> None:
    create_identity(client)
    response = client.post(
        "/api/v1/assessments/score",
        json={
            "user_id": "U001",
            "day": "2010-06-01",
            "model_version": "baseline.v1",
            "split": "VALIDATION",
            "feature": {
                "raw_score": 1.0,
                "supports": {"PERSON": {"active_days": 999}},
            },
        },
    )
    assert response.status_code == 422
    assert response.json()["code"] == "REQUEST_VALIDATION_ERROR"


def test_both_branches_missing_produces_no_score_and_no_alert(
    client: TestClient,
) -> None:
    create_identity(client)
    response = client.post(
        "/api/v1/assessments/score",
        json={
            "user_id": "U001",
            "day": "2010-06-01",
            "model_version": "empty.v1",
            "split": "VALIDATION",
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["state"] == "NO_SCORE"
    assert body["risk"] is None
    assert body["threshold"] is None
    assert body["is_alert"] is False
    assert client.get("/api/v1/alerts").json() == []


def test_assessment_lookup_requires_and_filters_config_version(
    client: TestClient,
) -> None:
    create_identity(client)
    base_payload = {
        "user_id": "U001",
        "day": "2010-06-01",
        "model_version": "same-model.v1",
        "split": "VALIDATION",
    }
    response = client.post(
        "/api/v1/assessments/score",
        json={**base_payload, "config_version": "framework.v4"},
    )
    assert response.status_code == 201, response.text

    missing_config = client.get(
        "/api/v1/assessments/U001/2010-06-01",
        params={"model_version": "same-model.v1"},
    )
    assert missing_config.status_code == 422

    selected = client.get(
        "/api/v1/assessments/U001/2010-06-01",
        params={
            "model_version": "same-model.v1",
            "config_version": "framework.v4",
        },
    )
    assert selected.status_code == 200, selected.text
    assert selected.json()["assessment"]["config_version"] == "framework.v4"

    absent = client.get(
        "/api/v1/assessments/U001/2010-06-01",
        params={
            "model_version": "same-model.v1",
            "config_version": "other-config.v1",
        },
    )
    assert absent.status_code == 404


def test_ingestion_idempotency_key_is_bound_to_one_manifest(
    client: TestClient,
) -> None:
    payload = {
        "source": "FILE",
        "input_uri": "file:///data/file-a.jsonl",
        "original_filename": "file-a.jsonl",
        "sha256": "a" * 64,
        "idempotency_key": "file-a-v1",
        "schema_version": "canonical-event.v1",
        "total_rows": 10,
    }
    first = client.post("/api/v1/ingestions", json=payload)
    assert first.status_code == 202, first.text

    retry = client.post("/api/v1/ingestions", json=payload)
    assert retry.status_code == 202, retry.text
    assert retry.json()["id"] == first.json()["id"]

    collision = client.post(
        "/api/v1/ingestions",
        json={**payload, "input_uri": "file:///data/different.jsonl"},
    )
    assert collision.status_code == 409
    assert collision.json()["code"] == "IDEMPOTENCY_KEY_REUSED"
    assert collision.json()["details"]["mismatched_fields"] == ["input_uri"]


def test_completed_ingestion_accepts_only_idempotent_retries(
    client: TestClient,
) -> None:
    create_identity(client)
    job = client.post(
        "/api/v1/ingestions",
        json={
            "source": "FILE",
            "input_uri": "file:///data/one-event.jsonl",
            "sha256": "b" * 64,
            "idempotency_key": "one-event-v1",
            "schema_version": "canonical-event.v1",
            "total_rows": 1,
        },
    )
    assert job.status_code == 202, job.text
    job_id = job.json()["id"]
    event = {
        "event_uid": "completed-job:1",
        "source": "FILE",
        "original_id": "completed-job:1",
        "timestamp": "2010-06-01T09:00:00+00:00",
        "user_id": "U001",
        "action": "FILE_COPY",
        "source_payload": {},
    }

    accepted = client.post(
        "/api/v1/events/batch",
        json={"ingestion_job_id": job_id, "events": [event]},
    )
    assert accepted.status_code == 202, accepted.text
    assert client.get(f"/api/v1/ingestions/{job_id}").json()["status"] == "COMPLETED"

    retry = client.post(
        "/api/v1/events/batch",
        json={"ingestion_job_id": job_id, "events": [event]},
    )
    assert retry.status_code == 202, retry.text
    assert retry.json()["duplicates"] == 1

    new_event = {
        **event,
        "event_uid": "completed-job:2",
        "original_id": "completed-job:2",
    }
    rejected = client.post(
        "/api/v1/events/batch",
        json={"ingestion_job_id": job_id, "events": [new_event]},
    )
    assert rejected.status_code == 409
    assert rejected.json()["code"] == "INGESTION_ALREADY_COMPLETED"


def test_scoring_release_is_idempotent_and_uses_server_config(
    client: TestClient,
) -> None:
    create_identity(client)
    payload = {
        "user_id": "U001",
        "day": "2010-06-01",
        "model_version": "idempotent.v1",
        "config_version": "framework.v4",
        "split": "VALIDATION",
    }
    first = client.post("/api/v1/assessments/score", json=payload)
    assert first.status_code == 201, first.text
    retry = client.post("/api/v1/assessments/score", json=payload)
    assert retry.status_code == 201, retry.text
    assert retry.json()["id"] == first.json()["id"]

    changed_input = client.post(
        "/api/v1/assessments/score",
        json={**payload, "feature": {"raw_score": 0.1}},
    )
    assert changed_input.status_code == 409
    assert changed_input.json()["code"] == "ASSESSMENT_RELEASE_CONFLICT"

    wrong_weights = client.post(
        "/api/v1/assessments/score",
        json={
            **payload,
            "model_version": "wrong-weights.v1",
            "feature_weight": 0.4,
            "sequence_weight": 0.6,
        },
    )
    assert wrong_weights.status_code == 422
    assert wrong_weights.json()["code"] == "FUSION_CONFIG_MISMATCH"

    wrong_threshold = client.post(
        "/api/v1/assessments/score",
        json={**payload, "model_version": "wrong-threshold.v1", "alert_threshold": 0.9},
    )
    assert wrong_threshold.status_code == 422
    assert wrong_threshold.json()["code"] == "ALERT_THRESHOLD_MISMATCH"

    wrong_config = client.post(
        "/api/v1/assessments/score",
        json={**payload, "model_version": "wrong-config.v1", "config_version": "framework.v1"},
    )
    assert wrong_config.status_code == 422
    assert wrong_config.json()["code"] == "CONFIG_VERSION_NOT_ACTIVE"
