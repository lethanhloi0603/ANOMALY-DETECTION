"""PostgreSQL-only concurrency coverage for immutable Personal releases."""

from __future__ import annotations

import copy
import os
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Barrier

import pytest
import sqlalchemy as sa
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker
from test_safe_update_service_flow import (
    CONFIG_VERSION,
    FIRST_PRODUCTION_DAY,
    MODEL_VERSION,
    _accepted_candidate,
    _identity,
    _role_parent,
    _source_day,
    _watermarks,
)

import app.services as services_module
from app.catalog import load_framework_config
from app.database import init_db, make_engine
from app.domain.safe_update import SafeUpdatePolicy
from app.models import (
    Organization,
    PersonalReferenceAccumulator,
    ReferenceLevel,
    ReferenceProfile,
    ReferenceRelease,
    ReferenceReleaseKind,
    SafeUpdateCandidate,
    UpdateStatus,
)
from app.schemas import BranchAssessmentInput
from app.services import _record_safe_update_admission, process_safe_updates

pytestmark = pytest.mark.postgres


@pytest.fixture
def postgres_engine() -> Iterator[Engine]:
    database_url = os.getenv("TEST_POSTGRES_URL")
    if not database_url:
        pytest.skip("TEST_POSTGRES_URL is not configured")

    admin_engine = make_engine(database_url)
    if admin_engine.dialect.name != "postgresql":
        admin_engine.dispose()
        pytest.fail("TEST_POSTGRES_URL must use PostgreSQL")

    schema_name = f"test_safe_update_{uuid.uuid4().hex}"
    quoted_schema = admin_engine.dialect.identifier_preparer.quote_identifier(
        schema_name
    )
    scoped_engine: Engine | None = None
    schema_created = False
    try:
        with admin_engine.begin() as connection:
            connection.exec_driver_sql(f"CREATE SCHEMA {quoted_schema}")
        schema_created = True

        scoped_engine = sa.create_engine(
            database_url,
            connect_args={"options": f"-csearch_path={schema_name}"},
            pool_pre_ping=True,
        )
        init_db(scoped_engine)
        yield scoped_engine
    finally:
        if scoped_engine is not None:
            scoped_engine.dispose()
        if schema_created:
            with admin_engine.begin() as connection:
                connection.exec_driver_sql(
                    f"DROP SCHEMA {quoted_schema} CASCADE"
                )
        admin_engine.dispose()


def test_two_workers_materialize_exactly_one_bootstrap_release(
    postgres_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    framework_config = copy.deepcopy(load_framework_config())
    framework_config["safe_personalized_update"]["release"][
        "materialization_enabled"
    ] = True
    monkeypatch.setattr(
        services_module,
        "load_framework_config",
        lambda: framework_config,
    )
    policy = SafeUpdatePolicy.from_framework(framework_config)

    factory = sessionmaker(
        bind=postgres_engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )
    with factory() as session:
        organization, user, assignment = _identity(
            session,
            external_user_id="U-PG-CONCURRENT",
        )
        parent = _role_parent(session, organization, assignment)
        _, first_score, first_assessment = _source_day(
            session,
            organization,
            user,
            assignment,
            parent,
            day=FIRST_PRODUCTION_DAY,
            raw_score=7.0,
        )
        first_candidate = _record_safe_update_admission(
            session,
            organization,
            user,
            assignment,
            assessment=first_assessment,
            branch_score=first_score,
            branch_input=BranchAssessmentInput(raw_score=7.0),
            profiles={ReferenceLevel.ROLE: parent},
            role_known=True,
            framework_config=framework_config,
            policy=policy,
            actor="postgres-setup",
            request_id="postgres-admission",
        )
        assert first_candidate is not None
        accumulator = session.get(
            PersonalReferenceAccumulator,
            first_candidate.accumulator_id,
        )
        assert accumulator is not None

        for index in range(1, 60):
            offset = (index * 89) // 59
            _accepted_candidate(
                session,
                organization,
                user,
                assignment,
                accumulator,
                parent,
                day=FIRST_PRODUCTION_DAY + timedelta(days=offset),
            )
        _watermarks(session, organization, start_offset=0, end_offset=119)
        session.commit()

        organization_id = organization.id
        accumulator_id = accumulator.id
        parent_id = parent.id
        parent_checksum = parent.checksum

    start = Barrier(2)

    def run_worker(worker_number: int) -> dict[str, int]:
        with factory() as session:
            organization = session.get(Organization, organization_id)
            assert organization is not None
            session.execute(sa.text("SET LOCAL lock_timeout = '10s'"))
            session.execute(sa.text("SET LOCAL statement_timeout = '30s'"))
            start.wait(timeout=15)
            result = process_safe_updates(
                session,
                organization,
                model_version=MODEL_VERSION,
                config_version=CONFIG_VERSION,
                runtime_materialization_enabled=True,
                limit=1_000,
                actor=f"postgres-worker-{worker_number}",
                request_id=f"postgres-worker-{worker_number}",
            )
            session.commit()
            return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_worker, worker) for worker in (1, 2)]
        results = [future.result(timeout=45) for future in futures]

    assert sum(result["accepted"] for result in results) == 1
    assert sum(result["applied"] for result in results) == 60

    with factory() as session:
        accumulator = session.get(PersonalReferenceAccumulator, accumulator_id)
        parent = session.get(ReferenceProfile, parent_id)
        releases = list(
            session.scalars(
                select(ReferenceRelease).where(
                    ReferenceRelease.accumulator_id == accumulator_id
                )
            )
        )
        children = list(
            session.scalars(
                select(ReferenceProfile).where(
                    ReferenceProfile.organization_id == organization_id,
                    ReferenceProfile.level == ReferenceLevel.PERSON,
                    ReferenceProfile.release_kind
                    == ReferenceReleaseKind.BOOTSTRAP,
                )
            )
        )
        applied_count = session.scalar(
            select(sa.func.count(SafeUpdateCandidate.id)).where(
                SafeUpdateCandidate.accumulator_id == accumulator_id,
                SafeUpdateCandidate.status == UpdateStatus.APPLIED,
            )
        )

        assert accumulator is not None
        assert parent is not None
        assert len(releases) == 1
        assert len(children) == 1
        release = releases[0]
        child = children[0]
        assert release.release_sequence == 1
        assert {item.release_sequence for item in releases} == {1}
        assert release.applied_candidate_count == 60
        assert release.child_reference_profile_id == child.id
        assert accumulator.release_sequence == 1
        assert accumulator.active_reference_profile_id == child.id
        assert applied_count == 60
        assert parent.checksum == parent_checksum
