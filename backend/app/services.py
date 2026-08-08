from __future__ import annotations

import hashlib
import math
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import Session

from app.catalog import (
    FEATURE_SCHEMA_VERSION,
    SEQUENCE_SCHEMA_VERSION,
    load_feature_catalog,
    load_framework_config,
    load_sequence_config,
)
from app.domain.fusion import fuse_scores
from app.domain.readiness import (
    FeatureGlobalSupport,
    FeaturePersonSupport,
    FeatureRoleSupport,
    ReadinessDecision,
    SequenceGlobalSupport,
    SequencePersonSupport,
    SequenceRoleSupport,
    select_feature_reference,
    select_sequence_reference,
)
from app.domain.safe_update import (
    FeatureBootstrapSupport,
    ReleaseWindowEntry,
    SafeUpdatePolicy,
    SequenceBootstrapSupport,
    admission_allowed,
    bounded_release_capacity,
    build_personal_calibrator,
    empirical_percentile,
    feature_bootstrap_readiness,
    quarantine_window,
    reference_percentile,
    robust_score_statistics,
    sequence_bootstrap_readiness,
)
from app.errors import ApiError, conflict, not_found
from app.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
    Artifact,
    ArtifactKind,
    ArtifactStatus,
    AssessmentStatus,
    AuditLog,
    Branch,
    BranchScore,
    CanonicalEvent,
    DataSplit,
    EventSource,
    FeatureCatalog,
    IngestionCheckpoint,
    IngestionJob,
    IngestSource,
    JobStatus,
    LegacyTimeContext,
    Organization,
    PCContext,
    PersonalAccumulatorStatus,
    PersonalReferenceAccumulator,
    ReferenceLevel,
    ReferenceProfile,
    ReferenceRelease,
    ReferenceReleaseKind,
    RiskAssessment,
    Role,
    RoleAssignment,
    SafeUpdateCandidate,
    ScoreStatus,
    ScoringWatermark,
    UpdateStatus,
    User,
    UserDayFeature,
    UserDaySequence,
    UserStatus,
    utc_now,
)
from app.schemas import (
    AlertPatch,
    ArtifactCreate,
    AssessmentCreate,
    BranchAssessmentInput,
    CanonicalEventIn,
    CheckpointUpsert,
    EventBatchCreate,
    FeatureVectorUpsert,
    IngestionJobCreate,
    ReferenceProfileCreate,
    RoleAssignmentCreate,
    RoleCreate,
    SequenceUpsert,
    TimestampBasis,
    UserCreate,
)
from app.validation import (
    DomainValidationError,
    dataset_split,
    enforce_cert_http_payload,
    enforce_label_firewall,
    enforce_metadata_only_payload,
    risk_severity,
    sha256_json,
    truncate_sequence,
    validate_feature_vector,
)

DEFAULT_ORGANIZATION_SLUG = "default"


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _lock_assessment_release(
    session: Session,
    *,
    organization_id: uuid.UUID,
    user_id: uuid.UUID,
    day: date,
    model_version: str,
    config_version: str,
) -> None:
    """Serialize one assessment release on PostgreSQL.

    SQLite remains the development/test database and serializes writers at the
    database level. PostgreSQL needs an explicit transaction-scoped lock so two
    scorer workers cannot both create branch scores before the unique release
    constraint is observed.
    """

    bind = session.get_bind()
    if bind.dialect.name != "postgresql":
        return
    payload = "|".join(
        (
            str(organization_id),
            str(user_id),
            day.isoformat(),
            model_version,
            config_version,
        )
    )
    raw_key = int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8],
        "big",
        signed=False,
    )
    signed_key = raw_key if raw_key < 2**63 else raw_key - 2**64
    session.execute(
        text("SELECT pg_advisory_xact_lock(:key)"),
        {"key": signed_key},
    )


def ensure_default_organization(session: Session) -> Organization:
    organization = session.scalar(
        select(Organization).where(Organization.slug == DEFAULT_ORGANIZATION_SLUG)
    )
    if organization is None:
        organization = Organization(
            slug=DEFAULT_ORGANIZATION_SLUG,
            name="Default Organization",
            timezone="UTC",
        )
        session.add(organization)
        session.flush()
    return organization


def get_user(session: Session, organization: Organization, external_user_id: str) -> User:
    user = session.scalar(
        select(User).where(
            User.organization_id == organization.id,
            User.external_user_id == external_user_id,
        )
    )
    if user is None:
        raise not_found("user", external_user_id)
    return user


def get_role(session: Session, organization: Organization, role_code: str) -> Role:
    role = session.scalar(
        select(Role).where(
            Role.organization_id == organization.id,
            Role.code == role_code,
        )
    )
    if role is None:
        raise not_found("role", role_code)
    return role


def role_assignment_at(session: Session, user_id: uuid.UUID, day: date) -> RoleAssignment | None:
    return session.scalar(
        select(RoleAssignment)
        .where(
            RoleAssignment.user_id == user_id,
            RoleAssignment.valid_from <= day,
            or_(RoleAssignment.valid_to.is_(None), RoleAssignment.valid_to > day),
        )
        .order_by(RoleAssignment.valid_from.desc())
        .limit(1)
    )


def role_epoch_anchor(
    session: Session,
    assignment: RoleAssignment,
) -> RoleAssignment:
    """Collapse contiguous same-role LDAP rows into one Person-profile epoch."""

    anchor = assignment
    predecessors = session.scalars(
        select(RoleAssignment)
        .where(
            RoleAssignment.user_id == assignment.user_id,
            RoleAssignment.valid_from < assignment.valid_from,
        )
        .order_by(RoleAssignment.valid_from.desc())
    )
    for predecessor in predecessors:
        if (
            predecessor.role_id != anchor.role_id
            or predecessor.valid_to != anchor.valid_from
        ):
            break
        anchor = predecessor
    return anchor


def append_audit(
    session: Session,
    organization: Organization | None,
    *,
    actor: str,
    action: str,
    entity_type: str,
    entity_id: str,
    request_id: str | None,
    details: dict[str, Any] | None = None,
    before: Any | None = None,
    after: Any | None = None,
) -> AuditLog:
    if organization is not None:
        # Serialize each tenant's hash-chain head on databases that support row
        # locks. SQLite ignores FOR UPDATE but already serializes writers.
        session.scalar(
            select(Organization.id).where(Organization.id == organization.id).with_for_update()
        )
    previous = session.scalar(
        select(AuditLog)
        .where(AuditLog.organization_id == (organization.id if organization is not None else None))
        .order_by(AuditLog.occurred_at.desc(), AuditLog.id.desc())
        .limit(1)
    )
    occurred_at = utc_now()
    payload = {
        "previous_hash": previous.event_hash if previous else None,
        "actor": actor,
        "action": action,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "request_id": request_id,
        "details": details or {},
        "before_hash": sha256_json(before) if before is not None else None,
        "after_hash": sha256_json(after) if after is not None else None,
        "occurred_at": occurred_at.isoformat(),
    }
    audit = AuditLog(
        organization_id=organization.id if organization else None,
        actor=actor,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        request_id=request_id,
        before_hash=payload["before_hash"],
        after_hash=payload["after_hash"],
        previous_hash=payload["previous_hash"],
        event_hash=sha256_json(payload),
        details_json=details or {},
        occurred_at=occurred_at,
    )
    session.add(audit)
    return audit


def create_user(
    session: Session,
    organization: Organization,
    payload: UserCreate,
    *,
    actor: str,
    request_id: str,
) -> User:
    existing = session.scalar(
        select(User).where(
            User.organization_id == organization.id,
            User.external_user_id == payload.user_id,
        )
    )
    if existing is not None:
        raise conflict("USER_EXISTS", "user already exists", {"user_id": payload.user_id})
    user = User(
        organization_id=organization.id,
        external_user_id=payload.user_id,
        display_name=payload.display_name,
        status=UserStatus(payload.status.lower()),
        attributes_json=payload.metadata,
    )
    session.add(user)
    session.flush()
    append_audit(
        session,
        organization,
        actor=actor,
        action="user.created",
        entity_type="user",
        entity_id=str(user.id),
        request_id=request_id,
        after={"external_user_id": user.external_user_id, "status": user.status.value},
    )
    return user


def create_role(
    session: Session,
    organization: Organization,
    payload: RoleCreate,
    *,
    actor: str,
    request_id: str,
) -> Role:
    existing = session.scalar(
        select(Role).where(
            Role.organization_id == organization.id,
            Role.code == payload.role_code,
        )
    )
    if existing is not None:
        raise conflict("ROLE_EXISTS", "role already exists", {"role_code": payload.role_code})
    role = Role(
        organization_id=organization.id,
        code=payload.role_code,
        name=payload.display_name,
        family=payload.role_family,
        is_unknown=payload.role_code.upper() == "UNKNOWN",
        attributes_json=payload.metadata,
    )
    session.add(role)
    session.flush()
    append_audit(
        session,
        organization,
        actor=actor,
        action="role.created",
        entity_type="role",
        entity_id=str(role.id),
        request_id=request_id,
        after={"code": role.code, "family": role.family},
    )
    return role


def assign_role(
    session: Session,
    organization: Organization,
    user: User,
    payload: RoleAssignmentCreate,
    *,
    actor: str,
    request_id: str,
) -> RoleAssignment:
    role = get_role(session, organization, payload.role_code)
    closed_day_statement = select(ScoringWatermark.day).where(
        ScoringWatermark.organization_id == organization.id,
        ScoringWatermark.day >= payload.valid_from,
    )
    if payload.valid_to is not None:
        closed_day_statement = closed_day_statement.where(
            ScoringWatermark.day < payload.valid_to
        )
    closed_day = session.scalar(closed_day_statement.order_by(ScoringWatermark.day).limit(1))
    if closed_day is not None:
        raise conflict(
            "ROLE_TIMELINE_DAY_ALREADY_CLOSED",
            "the role interval overlaps an immutable scoring-universe watermark",
            {
                "user_id": user.external_user_id,
                "closed_day": closed_day.isoformat(),
            },
        )
    assignment = RoleAssignment(
        user_id=user.id,
        role_id=role.id,
        valid_from=payload.valid_from,
        valid_to=payload.valid_to,
        source_snapshot_date=payload.source_snapshot_date,
        source_snapshot=payload.source_snapshot_date.isoformat(),
    )
    session.add(assignment)
    try:
        session.flush()
    except ValueError as exc:
        raise conflict(
            "ROLE_ASSIGNMENT_OVERLAP",
            str(exc),
            {"user_id": user.external_user_id},
        ) from exc
    append_audit(
        session,
        organization,
        actor=actor,
        action="role_assignment.created",
        entity_type="role_assignment",
        entity_id=str(assignment.id),
        request_id=request_id,
        after={
            "user_id": user.external_user_id,
            "role_code": role.code,
            "valid_from": assignment.valid_from,
            "valid_to": assignment.valid_to,
        },
    )
    return assignment


def create_ingestion_job(
    session: Session,
    organization: Organization,
    payload: IngestionJobCreate,
    *,
    actor: str,
    request_id: str,
) -> IngestionJob:
    existing = session.scalar(
        select(IngestionJob).where(
            IngestionJob.organization_id == organization.id,
            IngestionJob.idempotency_key == payload.idempotency_key,
        )
    )
    if existing is not None:
        requested = {
            "source": payload.source.value.lower(),
            "input_uri": payload.input_uri,
            "original_filename": payload.original_filename,
            "sha256": payload.sha256.lower(),
            "schema_version": payload.schema_version,
            "total_rows": payload.total_rows,
        }
        persisted = {
            "source": existing.source.value,
            "input_uri": existing.input_uri,
            "original_filename": existing.original_filename,
            "sha256": existing.sha256,
            "schema_version": existing.schema_version,
            "total_rows": existing.total_rows,
        }
        mismatched = sorted(key for key in requested if requested[key] != persisted[key])
        if mismatched:
            raise conflict(
                "IDEMPOTENCY_KEY_REUSED",
                "idempotency_key is already bound to a different ingestion request",
                {"mismatched_fields": mismatched},
            )
        return existing

    existing_content = session.scalar(
        select(IngestionJob).where(
            IngestionJob.organization_id == organization.id,
            IngestionJob.source == IngestSource(payload.source.value.lower()),
            IngestionJob.sha256 == payload.sha256.lower(),
        )
    )
    if existing_content is not None:
        return existing_content

    job = IngestionJob(
        organization_id=organization.id,
        source=IngestSource(payload.source.value.lower()),
        status=JobStatus.PENDING,
        input_uri=payload.input_uri,
        original_filename=payload.original_filename,
        sha256=payload.sha256.lower(),
        idempotency_key=payload.idempotency_key,
        schema_version=payload.schema_version,
        total_rows=payload.total_rows,
    )
    session.add(job)
    session.flush()
    append_audit(
        session,
        organization,
        actor=actor,
        action="ingestion.created",
        entity_type="ingestion_job",
        entity_id=str(job.id),
        request_id=request_id,
        after={"source": job.source.value, "sha256": job.sha256},
    )
    return job


def get_ingestion_job(
    session: Session,
    organization: Organization,
    job_id: str,
    *,
    for_update: bool = False,
) -> IngestionJob:
    try:
        parsed = uuid.UUID(job_id)
    except ValueError as exc:
        raise not_found("ingestion job", job_id) from exc
    statement = select(IngestionJob).where(
        IngestionJob.id == parsed,
        IngestionJob.organization_id == organization.id,
    )
    if for_update:
        statement = statement.with_for_update()
    job = session.scalar(statement)
    if job is None:
        raise not_found("ingestion job", job_id)
    return job


def upsert_checkpoint(
    session: Session,
    organization: Organization,
    job: IngestionJob,
    payload: CheckpointUpsert,
    *,
    actor: str,
    request_id: str,
) -> IngestionCheckpoint:
    locked_job = session.scalar(
        select(IngestionJob)
        .where(
            IngestionJob.id == job.id,
            IngestionJob.organization_id == organization.id,
        )
        .with_for_update()
    )
    if locked_job is None:
        raise not_found("ingestion job", str(job.id))
    if locked_job.status in {JobStatus.FAILED, JobStatus.CANCELLED}:
        raise conflict("INGESTION_NOT_WRITABLE", "ingestion job is not writable")
    if payload.input_checksum and payload.input_checksum.lower() != locked_job.sha256.lower():
        raise conflict(
            "CHECKPOINT_INPUT_MISMATCH",
            "checkpoint input checksum does not match the ingestion manifest",
        )

    checkpoint = session.scalar(
        select(IngestionCheckpoint)
        .where(
            IngestionCheckpoint.job_id == locked_job.id,
            IngestionCheckpoint.partition_key == payload.partition_key,
        )
        .with_for_update()
    )
    if checkpoint is None:
        checkpoint = IngestionCheckpoint(
            job_id=locked_job.id,
            partition_key=payload.partition_key,
        )
        session.add(checkpoint)
    elif payload.row_offset < checkpoint.row_offset:
        raise conflict(
            "CHECKPOINT_REGRESSION",
            "checkpoint row_offset cannot move backwards",
            {"current": checkpoint.row_offset, "requested": payload.row_offset},
        )
    elif (
        checkpoint.input_checksum
        and payload.input_checksum
        and checkpoint.input_checksum.lower() != payload.input_checksum.lower()
    ):
        raise conflict(
            "CHECKPOINT_INPUT_CHANGED",
            "checkpoint cannot be reused for different input content",
        )
    checkpoint.row_offset = payload.row_offset
    checkpoint.cursor_json = payload.cursor
    checkpoint.input_checksum = payload.input_checksum
    checkpoint.output_checksum = payload.output_checksum
    session.flush()
    append_audit(
        session,
        organization,
        actor=actor,
        action="ingestion.checkpoint",
        entity_type="ingestion_checkpoint",
        entity_id=str(checkpoint.id),
        request_id=request_id,
        after={"partition": checkpoint.partition_key, "row_offset": checkpoint.row_offset},
    )
    return checkpoint


def _implicit_ingestion_job(
    session: Session,
    organization: Organization,
    events: list[CanonicalEventIn],
) -> IngestionJob:
    sources = {event.source.value for event in events}
    if len(sources) != 1:
        raise DomainValidationError(
            "MIXED_SOURCE_BATCH",
            "one ingestion batch must contain exactly one source",
            {"sources": sorted(sources)},
        )
    digest = sha256_json([event.model_dump(mode="json") for event in events])
    source = IngestSource(next(iter(sources)).lower())
    job = session.scalar(
        select(IngestionJob)
        .where(
            IngestionJob.organization_id == organization.id,
            IngestionJob.idempotency_key == f"api:{digest}",
        )
        .with_for_update()
    )
    if job is None:
        job = IngestionJob(
            organization_id=organization.id,
            source=source,
            status=JobStatus.RUNNING,
            input_uri="api://events/batch",
            sha256=digest,
            idempotency_key=f"api:{digest}",
            schema_version="canonical-event.v1",
            total_rows=len(events),
            started_at=utc_now(),
        )
        session.add(job)
        session.flush()
    return job


def ingest_events(
    session: Session,
    organization: Organization,
    payload: EventBatchCreate,
    *,
    actor: str,
    request_id: str,
) -> tuple[int, int, list[str]]:
    if payload.ingestion_job_id:
        job = get_ingestion_job(
            session,
            organization,
            payload.ingestion_job_id,
            for_update=True,
        )
        if job.status in {JobStatus.FAILED, JobStatus.CANCELLED}:
            raise conflict("INGESTION_NOT_WRITABLE", "ingestion job is not writable")
    else:
        job = _implicit_ingestion_job(session, organization, payload.events)

    was_completed = job.status is JobStatus.COMPLETED
    inserted = 0
    duplicates = 0
    accepted_uids: list[str] = []
    pending_by_uid: dict[str, str] = {}
    pending_originals: dict[tuple[EventSource, str], str] = {}
    zone = ZoneInfo(organization.timezone)
    min_timestamp = job.min_event_timestamp
    max_timestamp = job.max_event_timestamp
    # SQLite returns timezone-aware columns as naive values. Treat persisted
    # ingestion bounds as UTC so later batches remain comparable.
    if min_timestamp is not None and min_timestamp.tzinfo is None:
        min_timestamp = min_timestamp.replace(tzinfo=UTC)
    if max_timestamp is not None and max_timestamp.tzinfo is None:
        max_timestamp = max_timestamp.replace(tzinfo=UTC)

    for row_number, event_in in enumerate(payload.events, start=1):
        if job.source.value != event_in.source.value.lower():
            raise DomainValidationError(
                "INGESTION_SOURCE_MISMATCH",
                "event source does not match ingestion job",
                {
                    "job_source": job.source.value,
                    "event_source": event_in.source.value,
                    "event_uid": event_in.event_uid,
                },
            )
        enforce_label_firewall(event_in.source_payload)
        enforce_metadata_only_payload(event_in.source_payload)
        if event_in.source.value == "HTTP":
            enforce_cert_http_payload(event_in.source_payload)
        user = get_user(session, organization, event_in.user_id)
        if event_in.timestamp_basis is TimestampBasis.CERT_LOCAL_WALL_CLOCK:
            # UTC is only a storage sentinel here. The CERT dataset supplies no
            # authoritative timezone, so preserve its wall-clock fields exactly.
            timestamp = event_in.timestamp.replace(tzinfo=None).replace(tzinfo=UTC)
            event_date = timestamp.date()
        else:
            timestamp = event_in.timestamp.astimezone(UTC)
            event_date = timestamp.astimezone(zone).date()
        fingerprint_payload = {
            **event_in.model_dump(mode="json"),
            "timestamp": timestamp.isoformat(),
            "event_date": event_date.isoformat(),
        }
        payload_hash = sha256_json(fingerprint_payload)
        pending_hash = pending_by_uid.get(event_in.event_uid)
        if pending_hash is not None:
            if pending_hash != payload_hash:
                raise conflict(
                    "EVENT_UID_COLLISION",
                    "event_uid is repeated in the batch with different content",
                    {"event_uid": event_in.event_uid},
                )
            duplicates += 1
            accepted_uids.append(event_in.event_uid)
            continue

        event_source = EventSource(event_in.source.value.lower())
        original_key = (event_source, event_in.original_id)
        pending_uid = pending_originals.get(original_key)
        if pending_uid is not None:
            raise conflict(
                "ORIGINAL_ID_COLLISION",
                "source/original_id is repeated with another event_uid in the batch",
                {
                    "original_id": event_in.original_id,
                    "source": event_in.source.value,
                    "event_uids": [pending_uid, event_in.event_uid],
                },
            )

        existing = session.scalar(
            select(CanonicalEvent).where(
                CanonicalEvent.organization_id == organization.id,
                CanonicalEvent.event_uid == event_in.event_uid,
            )
        )
        if existing is not None:
            if existing.payload_hash != payload_hash:
                raise conflict(
                    "EVENT_UID_COLLISION",
                    "event_uid already exists with different content",
                    {"event_uid": event_in.event_uid},
                )
            duplicates += 1
            accepted_uids.append(event_in.event_uid)
            continue
        same_original = session.scalar(
            select(CanonicalEvent).where(
                CanonicalEvent.organization_id == organization.id,
                CanonicalEvent.source == event_source,
                CanonicalEvent.original_id == event_in.original_id,
            )
        )
        if same_original is not None:
            raise conflict(
                "ORIGINAL_ID_COLLISION",
                "source/original_id already maps to another event_uid",
                {"original_id": event_in.original_id, "source": event_in.source.value},
            )

        action = event_in.action.upper()
        if event_in.source.value == "FILE":
            action = "FILE_COPY"
        event = CanonicalEvent(
            organization_id=organization.id,
            event_uid=event_in.event_uid,
            source=event_source,
            original_id=event_in.original_id,
            event_timestamp=timestamp,
            event_date=event_date,
            user_id=user.id,
            pc=event_in.pc,
            action=action,
            object_ref=event_in.object,
            pc_context=PCContext.UNKNOWN,
            # The initial schema requires this legacy column. It now stores only
            # a date-derived calendar bucket; no fixed business-hour rule is used.
            legacy_time_context=(
                LegacyTimeContext.WEEKEND
                if event_date.weekday() >= 5
                else LegacyTimeContext.WORK
            ),
            source_payload={
                **event_in.source_payload,
                "_timestamp_basis": event_in.timestamp_basis.value,
            },
            payload_hash=payload_hash,
            ingestion_job_id=job.id,
            row_number=row_number,
        )
        session.add(event)
        pending_by_uid[event.event_uid] = payload_hash
        pending_originals[original_key] = event.event_uid
        inserted += 1
        accepted_uids.append(event.event_uid)
        user.first_seen = min(filter(None, [user.first_seen, event_date]), default=event_date)
        user.last_seen = max(filter(None, [user.last_seen, event_date]), default=event_date)
        min_timestamp = min(filter(None, [min_timestamp, timestamp]), default=timestamp)
        max_timestamp = max(filter(None, [max_timestamp, timestamp]), default=timestamp)

    if was_completed and inserted:
        raise conflict(
            "INGESTION_ALREADY_COMPLETED",
            "a completed ingestion job only accepts idempotent retries",
            {"new_event_count": inserted},
        )

    # Count accepted source rows, but cap at the audited manifest so retries do
    # not advance a completed job twice.
    accepted_rows = inserted + duplicates
    if not was_completed:
        job.processed_rows = (
            min(job.total_rows, job.processed_rows + accepted_rows)
            if job.total_rows > 0
            else job.processed_rows + accepted_rows
        )
    job.min_event_timestamp = min_timestamp
    job.max_event_timestamp = max_timestamp
    job.progress_json = {
        **job.progress_json,
        "last_batch_inserted": inserted,
        "last_batch_duplicates": duplicates,
    }
    if job.total_rows == 0 and not was_completed:
        job.total_rows = job.processed_rows
    if not was_completed and job.processed_rows + job.rejected_rows >= job.total_rows:
        job.status = JobStatus.COMPLETED
        job.finished_at = utc_now()
    elif job.status is JobStatus.PENDING:
        job.status = JobStatus.RUNNING
        job.started_at = utc_now()
    session.flush()
    append_audit(
        session,
        organization,
        actor=actor,
        action="events.ingested",
        entity_type="ingestion_job",
        entity_id=str(job.id),
        request_id=request_id,
        details={"inserted": inserted, "duplicates": duplicates},
    )
    return inserted, duplicates, accepted_uids


def list_events(
    session: Session,
    organization: Organization,
    *,
    external_user_id: str | None,
    day: date | None,
    source: str | None,
    limit: int,
    offset: int,
) -> list[tuple[CanonicalEvent, str]]:
    statement = (
        select(CanonicalEvent, User.external_user_id)
        .join(User, User.id == CanonicalEvent.user_id)
        .where(CanonicalEvent.organization_id == organization.id)
    )
    if external_user_id:
        statement = statement.where(User.external_user_id == external_user_id)
    if day:
        statement = statement.where(CanonicalEvent.event_date == day)
    if source:
        statement = statement.where(CanonicalEvent.source == EventSource(source.lower()))
    return list(
        session.execute(
            statement.order_by(CanonicalEvent.event_timestamp, CanonicalEvent.event_uid)
            .offset(offset)
            .limit(limit)
        ).all()
    )


def _feature_catalog(
    session: Session,
    schema_version: str = FEATURE_SCHEMA_VERSION,
) -> FeatureCatalog:
    catalog = session.scalar(
        select(FeatureCatalog).where(
            or_(
                FeatureCatalog.name == schema_version,
                FeatureCatalog.version == schema_version,
            )
        )
    )
    if catalog is None:
        raise ApiError(
            503,
            "FEATURE_CATALOG_NOT_SEEDED",
            f"feature catalog {schema_version!r} is not initialized",
        )
    return catalog


def upsert_feature_vector(
    session: Session,
    organization: Organization,
    user: User,
    day: date,
    payload: FeatureVectorUpsert,
    *,
    actor: str,
    request_id: str,
) -> UserDayFeature:
    enforce_label_firewall(payload.context)
    enforce_metadata_only_payload(payload.context)
    catalog = _feature_catalog(session, payload.schema_version)
    definitions = sorted(catalog.definitions, key=lambda item: item.ordinal)
    names = {definition.name for definition in definitions}
    validate_feature_vector(payload.values, payload.masks, names)
    values = [payload.values[item.name] for item in definitions]
    present_mask = [
        payload.masks.get(item.name, payload.values[item.name] is not None) for item in definitions
    ]
    checksum = payload.input_checksum or sha256_json(
        {"values": values, "present_mask": present_mask, "context": payload.context}
    )
    assignment = role_assignment_at(session, user.id, day)
    vector = session.scalar(
        select(UserDayFeature).where(
            UserDayFeature.organization_id == organization.id,
            UserDayFeature.user_id == user.id,
            UserDayFeature.day == day,
            UserDayFeature.catalog_id == catalog.id,
        )
    )
    if vector is None:
        vector = UserDayFeature(
            organization_id=organization.id,
            user_id=user.id,
            day=day,
            catalog_id=catalog.id,
            split=DataSplit(dataset_split(day).lower()),
        )
        session.add(vector)
    vector.role_assignment_id = assignment.id if assignment else None
    vector.values = values
    vector.present_mask = present_mask
    vector.is_observed_day = True
    has_canonical_event = (
        session.scalar(
            select(CanonicalEvent.id)
            .where(
                CanonicalEvent.organization_id == organization.id,
                CanonicalEvent.user_id == user.id,
                CanonicalEvent.event_date == day,
            )
            .limit(1)
        )
        is not None
    )
    total_event_count = payload.values.get("total_event_count")
    vector.is_active_day = has_canonical_event or bool(
        total_event_count is not None and total_event_count > 0
    )
    vector.context_json = payload.context
    vector.input_checksum = checksum
    session.flush()
    append_audit(
        session,
        organization,
        actor=actor,
        action="user_day_feature.upserted",
        entity_type="user_day_feature",
        entity_id=str(vector.id),
        request_id=request_id,
        details={"user_id": user.external_user_id, "day": day.isoformat()},
        after={"input_checksum": checksum, "schema": catalog.name},
    )
    return vector


def get_feature_vector(
    session: Session,
    organization: Organization,
    user: User,
    day: date,
    schema_version: str = FEATURE_SCHEMA_VERSION,
) -> tuple[UserDayFeature, dict[str, float | None], dict[str, bool]]:
    catalog = _feature_catalog(session, schema_version)
    vector = session.scalar(
        select(UserDayFeature).where(
            UserDayFeature.organization_id == organization.id,
            UserDayFeature.user_id == user.id,
            UserDayFeature.day == day,
            UserDayFeature.catalog_id == catalog.id,
        )
    )
    if vector is None:
        raise not_found("feature vector", f"{user.external_user_id}/{day}")
    definitions = sorted(catalog.definitions, key=lambda item: item.ordinal)
    values = {definition.name: vector.values[index] for index, definition in enumerate(definitions)}
    masks = {
        definition.name: bool(vector.present_mask[index])
        for index, definition in enumerate(definitions)
    }
    return vector, values, masks


_SEQUENCE_EVENT_LOOKUP_CHUNK_SIZE = 500
_CANONICAL_SOURCE_RANK = {
    EventSource.LOGON: 0,
    EventSource.DEVICE: 1,
    EventSource.FILE: 2,
    EventSource.HTTP: 3,
    EventSource.EMAIL: 4,
}


def _sequence_token_for_event(event: CanonicalEvent) -> str:
    if event.source is EventSource.LOGON:
        action = event.action.upper()
        if action in {"LOGON", "LOGOFF"}:
            return action
        raise DomainValidationError(
            "SEQUENCE_EVENT_ACTION_UNSUPPORTED",
            "LOGON source events must use LOGON or LOGOFF action",
            {"event_uid": event.event_uid, "action": event.action},
        )
    if event.source is EventSource.DEVICE:
        action = event.action.upper()
        if action == "CONNECT":
            return "DEVICE_CONNECT"
        if action == "DISCONNECT":
            return "DEVICE_DISCONNECT"
        raise DomainValidationError(
            "SEQUENCE_EVENT_ACTION_UNSUPPORTED",
            "DEVICE source events must use CONNECT or DISCONNECT action",
            {"event_uid": event.event_uid, "action": event.action},
        )
    if event.source is EventSource.FILE:
        return "FILE"
    if event.source is EventSource.HTTP:
        return "HTTP"
    if event.source is EventSource.EMAIL:
        return "EMAIL"
    raise DomainValidationError(
        "SEQUENCE_EVENT_SOURCE_UNSUPPORTED",
        "canonical event source cannot be mapped to the sequence vocabulary",
        {"event_uid": event.event_uid, "source": str(event.source)},
    )


def _validate_sequence_events(
    session: Session,
    organization: Organization,
    user: User,
    day: date,
    event_uids: list[str],
    tokens: list[str],
) -> list[CanonicalEvent]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for event_uid in event_uids:
        if event_uid in seen:
            duplicates.add(event_uid)
        seen.add(event_uid)
    if duplicates:
        raise DomainValidationError(
            "SEQUENCE_EVENT_UID_DUPLICATE",
            "sequence event_uids must be unique",
            {"event_uids": sorted(duplicates)[:20]},
        )

    events: list[CanonicalEvent] = []
    for start in range(0, len(event_uids), _SEQUENCE_EVENT_LOOKUP_CHUNK_SIZE):
        chunk = event_uids[start : start + _SEQUENCE_EVENT_LOOKUP_CHUNK_SIZE]
        events.extend(
            session.scalars(
                select(CanonicalEvent).where(
                    CanonicalEvent.organization_id == organization.id,
                    CanonicalEvent.event_uid.in_(chunk),
                )
            )
        )
    by_uid = {event.event_uid: event for event in events}
    missing = [event_uid for event_uid in event_uids if event_uid not in by_uid]
    if missing:
        raise DomainValidationError(
            "SEQUENCE_EVENT_UID_NOT_FOUND",
            "every sequence event_uid must exist in canonical events",
            {"event_uids": missing[:20], "count": len(missing)},
        )

    submitted_events = [by_uid[event_uid] for event_uid in event_uids]
    wrong_context = [
        event.event_uid
        for event in submitted_events
        if event.user_id != user.id or event.event_date != day
    ]
    if wrong_context:
        raise DomainValidationError(
            "SEQUENCE_EVENT_CONTEXT_MISMATCH",
            "sequence events must belong to the same organization, user, and day",
            {
                "event_uids": wrong_context[:20],
                "count": len(wrong_context),
                "user_id": user.external_user_id,
                "day": day.isoformat(),
            },
        )

    expected_events = sorted(
        submitted_events,
        key=lambda event: (
            event.event_timestamp,
            _CANONICAL_SOURCE_RANK[event.source],
            event.event_uid,
        ),
    )
    expected_uids = [event.event_uid for event in expected_events]
    if event_uids != expected_uids:
        first_difference = next(
            index
            for index, (actual, expected) in enumerate(zip(event_uids, expected_uids, strict=True))
            if actual != expected
        )
        raise DomainValidationError(
            "SEQUENCE_EVENT_ORDER_INVALID",
            "sequence event_uids must follow canonical deterministic order",
            {
                "first_difference": first_difference,
                "actual": event_uids[first_difference],
                "expected": expected_uids[first_difference],
            },
        )

    token_errors: list[dict[str, Any]] = []
    for index, (event, actual_token) in enumerate(zip(submitted_events, tokens, strict=True)):
        expected_token = _sequence_token_for_event(event)
        if actual_token != expected_token:
            token_errors.append(
                {
                    "index": index,
                    "event_uid": event.event_uid,
                    "source": event.source.value,
                    "action": event.action,
                    "actual": actual_token,
                    "expected": expected_token,
                }
            )
    if token_errors:
        raise DomainValidationError(
            "SEQUENCE_TOKEN_SOURCE_MISMATCH",
            "sequence tokens must match canonical event source/action mapping",
            {"events": token_errors[:20], "count": len(token_errors)},
        )
    return submitted_events


def _event_temporal_fields(
    event: CanonicalEvent,
    zone: ZoneInfo,
) -> tuple[str, dict[str, float | str]]:
    timestamp = event.event_timestamp
    timestamp_basis = event.source_payload.get(
        "_timestamp_basis",
        TimestampBasis.ABSOLUTE_OFFSET.value,
    )
    if timestamp_basis == TimestampBasis.CERT_LOCAL_WALL_CLOCK.value:
        local_timestamp = timestamp.replace(tzinfo=None)
    else:
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        local_timestamp = timestamp.astimezone(zone)
    minute_of_day = (
        local_timestamp.hour * 60
        + local_timestamp.minute
        + local_timestamp.second / 60.0
        + local_timestamp.microsecond / 60_000_000.0
    )
    angle = 2.0 * math.pi * minute_of_day / (24.0 * 60.0)
    calendar_context = "WEEKEND" if local_timestamp.weekday() >= 5 else "WEEKDAY"
    return calendar_context, {
        "time_sin": math.sin(angle),
        "time_cos": math.cos(angle),
        "calendar_context": calendar_context,
    }


def _train_pc_context_map(
    session: Session,
    organization: Organization,
) -> dict[str, tuple[bool, str | None]]:
    """Fit the frozen PC map from distinct Train user-days."""

    rows = session.execute(
        select(
            CanonicalEvent.pc,
            User.external_user_id,
            CanonicalEvent.event_date,
        )
        .join(User, CanonicalEvent.user_id == User.id)
        .where(
            CanonicalEvent.organization_id == organization.id,
            CanonicalEvent.event_date >= date(2010, 1, 2),
            CanonicalEvent.event_date <= date(2010, 5, 31),
            CanonicalEvent.pc.is_not(None),
        )
    )
    user_days_by_pc: dict[str, set[tuple[str, date]]] = defaultdict(set)
    for pc, external_user_id, event_day in rows:
        if pc and external_user_id:
            user_days_by_pc[str(pc)].add((str(external_user_id), event_day))

    result: dict[str, tuple[bool, str | None]] = {}
    for pc, user_days in user_days_by_pc.items():
        counts = Counter(user_id for user_id, _day in user_days)
        total = sum(counts.values())
        highest = max(counts.values())
        dominance = highest / total
        is_shared = len(counts) >= 5 and dominance < 0.5
        owners = sorted(user_id for user_id, count in counts.items() if count == highest)
        owner = owners[0] if not is_shared and dominance >= 0.5 else None
        result[pc] = (is_shared, owner)
    return result


def _computed_pc_contexts(
    session: Session,
    organization: Organization,
    user: User,
    events: list[CanonicalEvent],
) -> list[str]:
    profiles = _train_pc_context_map(session, organization)
    contexts: list[str] = []
    for event in events:
        profile = profiles.get(event.pc or "")
        if profile is None:
            contexts.append(PCContext.UNKNOWN.value.upper())
            continue
        is_shared, owner = profile
        if is_shared:
            contexts.append(PCContext.SHARED.value.upper())
        elif owner is None:
            contexts.append(PCContext.UNKNOWN.value.upper())
        elif owner == user.external_user_id:
            contexts.append(PCContext.OWN.value.upper())
        else:
            contexts.append(PCContext.FOREIGN.value.upper())
    return contexts


def _gap_bucket(minutes: float) -> str:
    if minutes < 1:
        return "0-1"
    if minutes < 5:
        return "1-5"
    if minutes < 30:
        return "5-30"
    if minutes <= 120:
        return "30-120"
    return ">120"


def _computed_gap_buckets(events: list[CanonicalEvent]) -> list[str | None]:
    gaps: list[str | None] = [None] if events else []
    for left, right in zip(events, events[1:], strict=False):
        left_timestamp = left.event_timestamp
        right_timestamp = right.event_timestamp
        minutes = max(0.0, (right_timestamp - left_timestamp).total_seconds() / 60.0)
        gaps.append(_gap_bucket(minutes))
    return gaps


def upsert_sequence(
    session: Session,
    organization: Organization,
    user: User,
    day: date,
    payload: SequenceUpsert,
    *,
    actor: str,
    request_id: str,
) -> UserDaySequence:
    enforce_label_firewall({"side_fields": payload.side_fields})
    enforce_metadata_only_payload({"side_fields": payload.side_fields})
    seq_len = len(payload.tokens)
    token_values = [item.value for item in payload.tokens]
    submitted_events = _validate_sequence_events(
        session,
        organization,
        user,
        day,
        payload.event_uids,
        token_values,
    )
    pc_values = _computed_pc_contexts(session, organization, user, submitted_events)
    gap_values = _computed_gap_buckets(submitted_events)
    zone = ZoneInfo(organization.timezone)
    client_side_fields = payload.side_fields or [{} for _ in range(seq_len)]
    temporal_fields = [_event_temporal_fields(event, zone) for event in submitted_events]
    calendar_values = [calendar for calendar, _fields in temporal_fields]
    side_fields = [
        {**client_values, **computed_values}
        for client_values, (_calendar, computed_values) in zip(
            client_side_fields,
            temporal_fields,
            strict=True,
        )
    ]
    token_values, truncated = truncate_sequence(token_values)
    pc_values, _ = truncate_sequence(pc_values)
    calendar_values, _ = truncate_sequence(calendar_values)
    gaps, _ = truncate_sequence(gap_values)
    event_uids, _ = truncate_sequence(payload.event_uids)
    side_fields, _ = truncate_sequence(side_fields)
    checksum = payload.input_checksum or sha256_json(
        {
            "tokens": token_values,
            "pc_contexts": pc_values,
            "calendar_contexts": calendar_values,
            "gap_buckets": gaps,
            "event_uids": event_uids,
            "side_fields": side_fields,
            "seq_len": seq_len,
        }
    )
    assignment = role_assignment_at(session, user.id, day)
    sequence = session.scalar(
        select(UserDaySequence).where(
            UserDaySequence.organization_id == organization.id,
            UserDaySequence.user_id == user.id,
            UserDaySequence.day == day,
            UserDaySequence.vocabulary_version == payload.schema_version,
        )
    )
    if sequence is None:
        sequence = UserDaySequence(
            organization_id=organization.id,
            user_id=user.id,
            day=day,
            vocabulary_version=payload.schema_version,
            split=DataSplit(dataset_split(day).lower()),
        )
        session.add(sequence)
    sequence.role_assignment_id = assignment.id if assignment else None
    sequence.tokens = token_values
    sequence.pc_contexts = pc_values
    sequence.calendar_contexts = calendar_values
    sequence.gap_buckets = gaps
    sequence.event_uids = event_uids
    sequence.side_fields = side_fields
    sequence.seq_len = seq_len
    sequence.truncated = truncated
    sequence.input_checksum = checksum
    session.flush()
    append_audit(
        session,
        organization,
        actor=actor,
        action="user_day_sequence.upserted",
        entity_type="user_day_sequence",
        entity_id=str(sequence.id),
        request_id=request_id,
        details={
            "user_id": user.external_user_id,
            "day": day.isoformat(),
            "seq_len": seq_len,
            "truncated": truncated,
        },
        after={"input_checksum": checksum},
    )
    return sequence


def get_sequence(
    session: Session,
    organization: Organization,
    user: User,
    day: date,
    schema_version: str = SEQUENCE_SCHEMA_VERSION,
) -> UserDaySequence:
    sequence = session.scalar(
        select(UserDaySequence).where(
            UserDaySequence.organization_id == organization.id,
            UserDaySequence.user_id == user.id,
            UserDaySequence.day == day,
            UserDaySequence.vocabulary_version == schema_version,
        )
    )
    if sequence is None:
        raise not_found("sequence", f"{user.external_user_id}/{day}")
    return sequence


def create_artifact(
    session: Session,
    organization: Organization,
    payload: ArtifactCreate,
    *,
    actor: str,
    request_id: str,
) -> Artifact:
    enforce_label_firewall(payload.metadata)
    kind_map = {
        "00_events": ArtifactKind.CANONICAL_EVENTS,
        "01_feature128": ArtifactKind.FEATURES,
        "02_sequences": ArtifactKind.SEQUENCES,
        "03_role_context": ArtifactKind.ROLE_CONTEXT,
        "04_branch_scores": ArtifactKind.BRANCH_SCORES,
        "05_reference_tables": ArtifactKind.REFERENCE_PROFILE,
        "06_risk_alerts": ArtifactKind.RISK_ASSESSMENTS,
    }
    job_id = None
    if payload.ingestion_job_id:
        job_id = get_ingestion_job(session, organization, payload.ingestion_job_id).id
    artifact = Artifact(
        organization_id=organization.id,
        ingestion_job_id=job_id,
        kind=kind_map[payload.artifact_type],
        status=ArtifactStatus(
            payload.status.lower()
            .replace("building", "staged")
            .replace("failed", "invalid")
            .replace("superseded", "archived")
        ),
        uri=payload.uri,
        checksum=payload.checksum,
        schema_version=payload.schema_version,
        row_count=payload.row_count,
        min_date=payload.min_date,
        max_date=payload.max_date,
        manifest_json=payload.metadata,
    )
    session.add(artifact)
    session.flush()
    append_audit(
        session,
        organization,
        actor=actor,
        action="artifact.created",
        entity_type="artifact",
        entity_id=str(artifact.id),
        request_id=request_id,
        after={"kind": artifact.kind.value, "checksum": artifact.checksum},
    )
    return artifact


def create_reference_profile(
    session: Session,
    organization: Organization,
    payload: ReferenceProfileCreate,
    *,
    actor: str,
    request_id: str,
) -> ReferenceProfile:
    enforce_label_firewall(payload.support)
    enforce_label_firewall(payload.statistics)
    enforce_metadata_only_payload(payload.support)
    enforce_metadata_only_payload(payload.statistics)
    level = ReferenceLevel(payload.level.value.lower())
    user_id = None
    role_id = None
    assignment_id = None
    if level is ReferenceLevel.PERSON:
        user = get_user(session, organization, payload.scope_key.removeprefix("user:"))
        assignment = role_assignment_at(session, user.id, payload.as_of_date)
        if assignment is None:
            raise DomainValidationError(
                "PERSON_ROLE_EPOCH_MISSING",
                "PERSON reference requires a role assignment at as_of_date",
            )
        assignment = role_epoch_anchor(session, assignment)
        user_id = user.id
        assignment_id = assignment.id
        scope_key = f"person:{assignment.id}"
    elif level is ReferenceLevel.ROLE:
        role = get_role(session, organization, payload.scope_key.removeprefix("role:"))
        role_id = role.id
        scope_key = f"role:{role.id}"
    else:
        scope_key = "global"

    calibrator = dict(payload.statistics.get("calibrator") or {})
    if calibrator.get("method") != "empirical_cdf.v1":
        raise DomainValidationError(
            "CALIBRATOR_METHOD_INVALID",
            "primary references require calibrator.method='empirical_cdf.v1'",
        )
    sorted_scores = calibrator.get("sorted_scores")
    if (
        not isinstance(sorted_scores, list)
        or not sorted_scores
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in sorted_scores
        )
    ):
        raise DomainValidationError(
            "CALIBRATOR_SCORES_INVALID",
            "empirical calibrator requires a non-empty finite sorted_scores array",
        )
    normalized_scores = [float(value) for value in sorted_scores]
    if normalized_scores != sorted(normalized_scores):
        raise DomainValidationError(
            "CALIBRATOR_SCORES_UNSORTED",
            "empirical calibrator sorted_scores must be non-decreasing",
        )
    calibrator["sorted_scores"] = normalized_scores

    support = dict(payload.support)
    if (
        payload.config_version == "framework.v5"
        and level is ReferenceLevel.PERSON
        and payload.branch.value == "FEATURE"
    ):
        observation_counts = support.get("feature_observation_counts")
        if (
            not isinstance(observation_counts, list)
            or len(observation_counts) != 128
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in observation_counts
            )
        ):
            raise DomainValidationError(
                "PERSON_FEATURE_OBSERVATION_COUNTS_REQUIRED",
                "framework.v5 Personal Feature references require 128 "
                "non-negative observation counts",
            )
        support["feature_observation_counts"] = [
            int(value) for value in observation_counts
        ]
        observation_days = int(
            support.get("feature_observation_days")
            or support.get("observations")
            or support.get("support_days")
            or support.get("active_days")
            or 0
        )
        if observation_days < 1 or any(
            value > observation_days for value in observation_counts
        ):
            raise DomainValidationError(
                "PERSON_FEATURE_OBSERVATION_DAYS_INVALID",
                "feature observation counts must not exceed their positive "
                "observation-day denominator",
            )
        support["feature_observation_days"] = observation_days
        support["feature_dimension"] = 128
        support["coverage"] = sum(observation_counts) / (observation_days * 128)
        used_counts = [value for value in observation_counts if value > 0]
        support["min_feature_observations"] = (
            min(used_counts) if used_counts else 0
        )
    support["_as_of_date"] = payload.as_of_date.isoformat()
    support["_reference_version"] = payload.version
    profile = ReferenceProfile(
        organization_id=organization.id,
        branch=Branch(payload.branch.value.lower()),
        level=level,
        scope_key=scope_key,
        user_id=user_id,
        role_id=role_id,
        role_assignment_id=assignment_id,
        model_version=payload.model_version,
        config_version=payload.config_version,
        catalog_version=(
            FEATURE_SCHEMA_VERSION
            if payload.branch.value == "FEATURE"
            else SEQUENCE_SCHEMA_VERSION
        ),
        fitted_from=(
            date.fromisoformat(str(support["fitted_from"])) if support.get("fitted_from") else None
        ),
        fitted_through=payload.fitted_through,
        support_days=int(
            support.get("support_days")
            or support.get("user_days")
            or support.get("sequence_days")
            or support.get("active_days")
            or 0
        ),
        support_users=int(
            support.get("support_users") or support.get("users") or support.get("peer_users") or 0
        ),
        support_transitions=int(support.get("transitions") or 0),
        coverage=(float(support["coverage"]) if support.get("coverage") is not None else None),
        support_json=support,
        statistics_json=payload.statistics,
        calibrator_json=calibrator,
        is_frozen=payload.frozen,
        checksum=payload.checksum,
    )
    session.add(profile)
    session.flush()
    append_audit(
        session,
        organization,
        actor=actor,
        action="reference_profile.created",
        entity_type="reference_profile",
        entity_id=str(profile.id),
        request_id=request_id,
        after={
            "branch": profile.branch.value,
            "level": profile.level.value,
            "scope": profile.scope_key,
            "checksum": profile.checksum,
        },
    )
    return profile


def _field(data: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return default


def _stored_scope(
    profiles: dict[ReferenceLevel, ReferenceProfile],
    level: ReferenceLevel,
) -> dict[str, Any] | None:
    profile = profiles.get(level)
    if profile is None:
        return None
    return {
        **profile.support_json,
        "support_as_of": profile.fitted_through,
    }


def _stored_staleness_gap(
    support: dict[str, Any],
    *,
    gap_keys: Sequence[str],
    day_keys: Sequence[str],
) -> int:
    explicit = _field(support, *gap_keys)
    if explicit is not None:
        return int(explicit)
    last_day_raw = _field(support, *day_keys)
    if last_day_raw is None:
        raise KeyError(day_keys[0])
    last_day = (
        last_day_raw
        if isinstance(last_day_raw, date)
        else date.fromisoformat(str(last_day_raw))
    )
    return max((support["support_as_of"] - last_day).days, 0)


def _build_feature_supports(
    profiles: dict[ReferenceLevel, ReferenceProfile],
    role_known: bool,
) -> tuple[
    FeaturePersonSupport | None,
    FeatureRoleSupport | None,
    FeatureGlobalSupport | None,
]:
    person = _stored_scope(profiles, ReferenceLevel.PERSON)
    role = _stored_scope(profiles, ReferenceLevel.ROLE)
    global_data = _stored_scope(profiles, ReferenceLevel.GLOBAL)
    try:
        return (
            FeaturePersonSupport(
                support_as_of=person["support_as_of"],
                active_days=int(_field(person, "active_days", "active_days_before_d")),
                span_days=int(person["span_days"]),
                active_days_current_role=int(
                    _field(
                        person,
                        "active_days_current_role",
                        "active_days_in_current_role",
                        default=_field(person, "active_days", "active_days_before_d"),
                    )
                ),
                coverage=float(
                    _field(
                        person,
                        "coverage",
                        "feature_coverage",
                        "mean_feature_coverage",
                    )
                ),
                min_feature_observations=int(
                    _field(
                        person,
                        "min_feature_observations",
                        "min_observations_per_used_feature",
                        "support_per_feature",
                        "minimum_nonzero_feature_support",
                    )
                ),
                last_active_gap_days=_stored_staleness_gap(
                    person,
                    gap_keys=("last_active_gap_days", "last_active_gap"),
                    day_keys=("last_active_day",),
                ),
            )
            if person
            else None,
            FeatureRoleSupport(
                support_as_of=role["support_as_of"],
                role_known=role_known,
                peer_users=int(_field(role, "peer_users", "peer_users_excluding_subject")),
                peer_user_days=int(role["peer_user_days"]),
                recent_user_days=int(_field(role, "recent_user_days", "recent_30d_user_days")),
                coverage=float(_field(role, "coverage", "feature_coverage")),
                min_feature_observations=int(
                    _field(
                        role,
                        "min_feature_observations",
                        "min_support_per_feature",
                        "support_per_feature",
                    )
                ),
            )
            if role
            else None,
            FeatureGlobalSupport(
                support_as_of=global_data["support_as_of"],
                users=int(global_data["users"]),
                user_days=int(global_data["user_days"]),
                coverage=float(_field(global_data, "coverage", "feature_coverage")),
                train_fitted=bool(
                    _field(global_data, "train_fitted", "fit_train_only", default=True)
                ),
            )
            if global_data
            else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DomainValidationError(
            "FEATURE_SUPPORT_INVALID",
            "feature support payload is incomplete or has invalid types",
            {"error": str(exc)},
        ) from exc


def _build_sequence_supports(
    profiles: dict[ReferenceLevel, ReferenceProfile],
    role_known: bool,
) -> tuple[
    SequencePersonSupport | None,
    SequenceRoleSupport | None,
    SequenceGlobalSupport | None,
]:
    person = _stored_scope(profiles, ReferenceLevel.PERSON)
    role = _stored_scope(profiles, ReferenceLevel.ROLE)
    global_data = _stored_scope(profiles, ReferenceLevel.GLOBAL)
    try:
        return (
            SequencePersonSupport(
                support_as_of=person["support_as_of"],
                sequence_days=int(person["sequence_days"]),
                transitions=int(person["transitions"]),
                span_days=int(person["span_days"]),
                sequence_days_current_role=int(
                    _field(
                        person,
                        "sequence_days_current_role",
                        "sequence_days_in_current_role",
                        default=person.get("sequence_days"),
                    )
                ),
                last_active_gap_days=_stored_staleness_gap(
                    person,
                    gap_keys=(
                        "last_active_gap_days",
                        "stale_gap_days",
                        "stale_gap",
                    ),
                    day_keys=("last_sequence_day", "last_active_day"),
                ),
            )
            if person
            else None,
            SequenceRoleSupport(
                support_as_of=role["support_as_of"],
                role_known=role_known,
                peer_users=int(role["peer_users"]),
                sequence_days=int(role["sequence_days"]),
                transitions=int(role["transitions"]),
                recent_transitions=int(
                    _field(role, "recent_transitions", "recent_30d_transitions")
                ),
            )
            if role
            else None,
            SequenceGlobalSupport(
                support_as_of=global_data["support_as_of"],
                users=int(global_data["users"]),
                sequence_days=int(global_data["sequence_days"]),
                transitions=int(global_data["transitions"]),
                train_fitted=bool(
                    _field(global_data, "train_fitted", "fit_train_only", default=True)
                ),
            )
            if global_data
            else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DomainValidationError(
            "SEQUENCE_SUPPORT_INVALID",
            "sequence support payload is incomplete or has invalid types",
            {"error": str(exc)},
        ) from exc


def _catalog_version_for_branch(branch: Branch) -> str:
    return FEATURE_SCHEMA_VERSION if branch is Branch.FEATURE else SEQUENCE_SCHEMA_VERSION


def _personal_accumulator_for_scope(
    session: Session,
    organization: Organization,
    user: User,
    epoch_assignment: RoleAssignment,
    *,
    branch: Branch,
    model_version: str,
    config_version: str,
    lock: bool = False,
) -> PersonalReferenceAccumulator | None:
    statement = select(PersonalReferenceAccumulator).where(
        PersonalReferenceAccumulator.organization_id == organization.id,
        PersonalReferenceAccumulator.user_id == user.id,
        PersonalReferenceAccumulator.role_assignment_id == epoch_assignment.id,
        PersonalReferenceAccumulator.branch == branch,
        PersonalReferenceAccumulator.model_version == model_version,
        PersonalReferenceAccumulator.config_version == config_version,
        PersonalReferenceAccumulator.catalog_version
        == _catalog_version_for_branch(branch),
    )
    if lock:
        statement = statement.with_for_update()
    return session.scalar(statement)


def _profile_for_decision(
    session: Session,
    organization: Organization,
    user: User,
    assignment: RoleAssignment | None,
    *,
    branch: Branch,
    level: ReferenceLevel,
    model_version: str,
    config_version: str,
    fitted_through: date,
    data_split: str,
    explicit_ids: dict[str, str],
    required: bool = True,
) -> ReferenceProfile | None:
    accumulator: PersonalReferenceAccumulator | None = None
    if level is ReferenceLevel.PERSON and assignment is not None:
        accumulator = _personal_accumulator_for_scope(
            session,
            organization,
            user,
            role_epoch_anchor(session, assignment),
            branch=branch,
            model_version=model_version,
            config_version=config_version,
        )
    explicit = explicit_ids.get(level.value.upper()) or explicit_ids.get(level.value)
    if explicit:
        try:
            profile = session.get(ReferenceProfile, uuid.UUID(explicit))
        except ValueError as exc:
            raise DomainValidationError(
                "REFERENCE_ID_INVALID", "reference profile ID must be a UUID"
            ) from exc
        if profile is None:
            raise not_found("reference profile", explicit)
        candidates = [profile]
    elif (
        level is ReferenceLevel.PERSON
        and data_split == "PRODUCTION"
        and accumulator is not None
    ):
        profile = (
            session.get(ReferenceProfile, accumulator.active_reference_profile_id)
            if accumulator.active_reference_profile_id is not None
            else None
        )
        candidates = [profile] if profile is not None else []
    else:
        selection_cutoff = fitted_through
        if data_split in {"VALIDATION", "TEST"}:
            selection_cutoff = min(
                fitted_through,
                date.fromisoformat(
                    str(load_framework_config()["splits"]["train"]["end"])
                ),
            )
        statement = select(ReferenceProfile).where(
            ReferenceProfile.organization_id == organization.id,
            ReferenceProfile.branch == branch,
            ReferenceProfile.level == level,
            ReferenceProfile.model_version == model_version,
            ReferenceProfile.config_version == config_version,
            ReferenceProfile.catalog_version == _catalog_version_for_branch(branch),
            ReferenceProfile.fitted_through <= selection_cutoff,
        )
        if level is ReferenceLevel.PERSON:
            if assignment is None:
                if required:
                    raise DomainValidationError(
                        "PERSON_ROLE_EPOCH_MISSING",
                        "PERSON selection requires a current role assignment",
                    )
                return None
            epoch_assignment = role_epoch_anchor(session, assignment)
            statement = statement.where(
                ReferenceProfile.user_id == user.id,
                ReferenceProfile.role_assignment_id == epoch_assignment.id,
            )
        elif level is ReferenceLevel.ROLE:
            if assignment is None:
                if required:
                    raise DomainValidationError(
                        "ROLE_CONTEXT_MISSING", "ROLE selection requires a current role"
                    )
                return None
            statement = statement.where(ReferenceProfile.role_id == assignment.role_id)
        else:
            statement = statement.where(
                ReferenceProfile.user_id.is_(None),
                ReferenceProfile.role_id.is_(None),
            )
        candidates = list(
            session.scalars(statement.order_by(ReferenceProfile.fitted_through.desc()).limit(1))
        )
    if not candidates:
        if not required:
            return None
        raise DomainValidationError(
            "REFERENCE_PROFILE_REQUIRED",
            "no compatible stored reference profile exists for the selected level",
            {"branch": branch.value, "level": level.value},
        )
    profile = candidates[0]
    if (
        level is ReferenceLevel.PERSON
        and data_split == "PRODUCTION"
        and accumulator is not None
        and profile.id != accumulator.active_reference_profile_id
    ):
        raise DomainValidationError(
            "PERSON_REFERENCE_NOT_ACTIVE",
            "production scoring must use the accumulator's active immutable Personal reference",
            {
                "provided": str(profile.id),
                "active": (
                    str(accumulator.active_reference_profile_id)
                    if accumulator.active_reference_profile_id
                    else None
                ),
            },
        )
    if (
        profile.organization_id != organization.id
        or profile.branch is not branch
        or profile.level is not level
        or profile.model_version != model_version
        or profile.config_version != config_version
        or profile.catalog_version != _catalog_version_for_branch(branch)
        or profile.fitted_through > fitted_through
    ):
        raise DomainValidationError(
            "REFERENCE_PROFILE_MISMATCH",
            "stored reference profile does not match the assessment decision",
        )
    if level is ReferenceLevel.PERSON:
        if assignment is None:
            raise DomainValidationError(
                "PERSON_ROLE_EPOCH_MISSING",
                "PERSON selection requires a current role assignment",
            )
        epoch_assignment = role_epoch_anchor(session, assignment)
        if (
            profile.user_id != user.id
            or profile.role_assignment_id != epoch_assignment.id
        ):
            raise DomainValidationError(
                "REFERENCE_PROFILE_SCOPE_MISMATCH",
                "PERSON reference does not belong to the current user/role epoch",
            )
    elif level is ReferenceLevel.ROLE:
        if assignment is None or profile.role_id != assignment.role_id:
            raise DomainValidationError(
                "REFERENCE_PROFILE_SCOPE_MISMATCH",
                "ROLE reference does not belong to the current role",
            )
    elif (
        profile.user_id is not None
        or profile.role_id is not None
        or profile.role_assignment_id is not None
    ):
        raise DomainValidationError(
            "REFERENCE_PROFILE_SCOPE_MISMATCH",
            "GLOBAL reference must not carry person or role scope",
        )
    if data_split in {"VALIDATION", "TEST"}:
        if not profile.is_frozen:
            raise DomainValidationError(
                "REFERENCE_NOT_FROZEN",
                "all PERSON, ROLE, and GLOBAL evaluation references must be frozen",
            )
        train_end = date.fromisoformat(
            str(load_framework_config()["splits"]["train"]["end"])
        )
        if profile.fitted_through > train_end:
            raise DomainValidationError(
                "REFERENCE_NOT_TRAIN_ONLY",
                "all evaluation references must be fitted on Train only",
            )
    return profile


def _profiles_for_branch(
    session: Session,
    organization: Organization,
    user: User,
    assignment: RoleAssignment | None,
    *,
    branch: Branch,
    assessment: AssessmentCreate,
    branch_input: BranchAssessmentInput,
) -> dict[ReferenceLevel, ReferenceProfile]:
    cutoff = assessment.day - timedelta(days=1)
    profiles: dict[ReferenceLevel, ReferenceProfile] = {}
    for level in (
        ReferenceLevel.PERSON,
        ReferenceLevel.ROLE,
        ReferenceLevel.GLOBAL,
    ):
        profile = _profile_for_decision(
            session,
            organization,
            user,
            assignment,
            branch=branch,
            level=level,
            model_version=assessment.model_version,
            config_version=assessment.config_version,
            fitted_through=cutoff,
            data_split=assessment.split,
            explicit_ids=branch_input.reference_profile_ids,
            required=False,
        )
        if profile is not None:
            profiles[level] = profile
    return profiles


def _decision_snapshot(
    decision: ReadinessDecision,
    profiles: dict[ReferenceLevel, ReferenceProfile],
) -> dict[str, Any]:
    return {
        "support_source": "stored_reference_artifact",
        "profiles": {
            level.value.upper(): {
                "reference_profile_id": str(profile.id),
                "fitted_through": profile.fitted_through.isoformat(),
                "support": profile.support_json,
            }
            for level, profile in profiles.items()
        },
        "evaluations": [
            {
                "level": evaluation.level.value,
                "eligible": evaluation.eligible,
                "reason_codes": list(evaluation.reason_codes),
            }
            for evaluation in decision.evaluations
        ],
    }


def _profile_percentile(
    session: Session,
    profile: ReferenceProfile,
    raw_score: float,
) -> float:
    parent_scores: Sequence[float] | None = None
    if profile.calibrator_json.get("method") == "parent_shrunk_ecdf.v1":
        parent_id = profile.calibration_parent_profile_id
        parent = session.get(ReferenceProfile, parent_id) if parent_id else None
        expected_id = str(
            profile.calibrator_json.get("parent_reference_profile_id", "")
        )
        expected_checksum = str(
            profile.calibrator_json.get("parent_reference_checksum", "")
        )
        if (
            parent is None
            or str(parent.id) != expected_id
            or parent.checksum != expected_checksum
            or parent.level not in {ReferenceLevel.ROLE, ReferenceLevel.GLOBAL}
        ):
            raise DomainValidationError(
                "REFERENCE_CALIBRATION_PARENT_INVALID",
                "Personal calibration parent is missing or does not match its immutable checksum",
                {"reference_profile_id": str(profile.id)},
            )
        parent_scores = parent.calibrator_json.get("sorted_scores")
    try:
        return reference_percentile(
            raw_score,
            profile.calibrator_json,
            parent_sorted_scores=parent_scores,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DomainValidationError(
            "REFERENCE_CALIBRATOR_INVALID",
            "selected reference does not contain a supported empirical CDF calibrator",
            {"reference_profile_id": str(profile.id), "error": str(exc)},
        ) from exc


def _persist_branch_score(
    session: Session,
    organization: Organization,
    user: User,
    assignment: RoleAssignment | None,
    *,
    branch: Branch,
    payload: BranchAssessmentInput,
    decision: ReadinessDecision,
    profiles: dict[ReferenceLevel, ReferenceProfile],
    assessment: AssessmentCreate,
    scoring_run_id: str,
) -> BranchScore:
    if decision.can_score:
        level = ReferenceLevel(decision.selected_level.value.lower())
        if payload.raw_score is None:
            raise DomainValidationError(
                "SELECTED_SCORE_MISSING",
                "selected reference requires a raw score",
                {"branch": branch.value, "level": level.value},
            )
        profile = profiles.get(level)
        if profile is None:
            raise DomainValidationError(
                "REFERENCE_PROFILE_REQUIRED",
                "selected readiness level has no stored reference profile",
                {"branch": branch.value, "level": level.value},
            )
        q_score = _profile_percentile(session, profile, float(payload.raw_score))
        score = BranchScore(
            organization_id=organization.id,
            user_id=user.id,
            day=assessment.day,
            role_assignment_id=assignment.id if assignment else None,
            branch=branch,
            status=ScoreStatus.SCORED,
            selected_level=level,
            reference_profile_id=profile.id,
            raw_score=payload.raw_score,
            calibrated_score=q_score,
            support_snapshot=_decision_snapshot(decision, profiles),
            fallback_reasons=list(decision.reason_codes),
            evidence=payload.evidence,
            model_version=assessment.model_version,
            config_version=assessment.config_version,
            scoring_run_id=scoring_run_id,
        )
    else:
        score = BranchScore(
            organization_id=organization.id,
            user_id=user.id,
            day=assessment.day,
            role_assignment_id=assignment.id if assignment else None,
            branch=branch,
            status=ScoreStatus.NO_SCORE,
            support_snapshot=_decision_snapshot(decision, profiles),
            fallback_reasons=list(decision.reason_codes),
            evidence=payload.evidence,
            model_version=assessment.model_version,
            config_version=assessment.config_version,
            scoring_run_id=scoring_run_id,
        )
    session.add(score)
    session.flush()
    return score


def _lock_personal_accumulator_scope(
    session: Session,
    organization: Organization,
    user: User,
    epoch_assignment: RoleAssignment,
    *,
    branch: Branch,
    model_version: str,
    config_version: str,
) -> None:
    if session.get_bind().dialect.name != "postgresql":
        return
    lock_payload = "|".join(
        (
            str(organization.id),
            str(user.id),
            str(epoch_assignment.id),
            branch.value,
            model_version,
            config_version,
            _catalog_version_for_branch(branch),
        )
    )
    raw_key = int.from_bytes(
        hashlib.sha256(lock_payload.encode("utf-8")).digest()[:8],
        "big",
        signed=False,
    )
    signed_key = raw_key if raw_key < 2**63 else raw_key - 2**64
    session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": signed_key})


def _admission_parent_for_branch(
    *,
    branch: Branch,
    score_date: date,
    branch_input: BranchAssessmentInput,
    profiles: dict[ReferenceLevel, ReferenceProfile],
    role_known: bool,
    framework_config: Mapping[str, Any],
) -> tuple[ReferenceProfile | None, list[str]]:
    parent_profiles = {
        level: profile
        for level, profile in profiles.items()
        if level in {ReferenceLevel.ROLE, ReferenceLevel.GLOBAL}
    }
    if branch is Branch.FEATURE:
        _, role, global_support = _build_feature_supports(parent_profiles, role_known)
        decision = select_feature_reference(
            score_date=score_date,
            person=None,
            role=role,
            global_support=global_support,
            config=framework_config,
        )
    else:
        _, role, global_support = _build_sequence_supports(parent_profiles, role_known)
        decision = select_sequence_reference(
            score_date=score_date,
            seq_len=int(branch_input.seq_len or 0),
            person=None,
            role=role,
            global_support=global_support,
            config=framework_config,
        )
    if not decision.can_score:
        return None, ["ADMISSION_PARENT_NOT_READY", *decision.reason_codes]
    selected = ReferenceLevel(decision.selected_level.value.lower())
    if selected not in {ReferenceLevel.ROLE, ReferenceLevel.GLOBAL}:
        return None, ["ADMISSION_PARENT_MUST_BE_NON_PERSON"]
    parent = profiles.get(selected)
    if parent is None:
        return None, ["ADMISSION_PARENT_PROFILE_MISSING"]
    if not parent.is_frozen:
        return None, ["ADMISSION_PARENT_NOT_FROZEN"]
    if parent.calibrator_json.get("method") != "empirical_cdf.v1":
        return None, ["ADMISSION_PARENT_CALIBRATOR_INVALID"]
    return parent, []


def _candidate_source_observation(
    session: Session,
    organization: Organization,
    user: User,
    *,
    branch: Branch,
    candidate_day: date,
) -> tuple[UserDayFeature | UserDaySequence | None, dict[str, Any]]:
    if branch is Branch.FEATURE:
        feature = session.scalar(
            select(UserDayFeature)
            .join(FeatureCatalog, FeatureCatalog.id == UserDayFeature.catalog_id)
            .where(
                UserDayFeature.organization_id == organization.id,
                UserDayFeature.user_id == user.id,
                UserDayFeature.day == candidate_day,
                FeatureCatalog.version == FEATURE_SCHEMA_VERSION,
            )
        )
        if feature is None or not feature.is_active_day:
            return None, {}
        present_ordinals = [
            index + 1
            for index, present in enumerate(feature.present_mask)
            if bool(present)
        ]
        return feature, {
            "active_day": True,
            "feature_dimension": len(feature.present_mask),
            "present_ordinals": present_ordinals,
        }
    sequence = session.scalar(
        select(UserDaySequence).where(
            UserDaySequence.organization_id == organization.id,
            UserDaySequence.user_id == user.id,
            UserDaySequence.day == candidate_day,
            UserDaySequence.vocabulary_version == SEQUENCE_SCHEMA_VERSION,
        )
    )
    if sequence is None or sequence.seq_len < 2:
        return None, {}
    return sequence, {
        "sequence_day": True,
        "transitions": sequence.seq_len - 1,
    }


def _get_or_create_personal_accumulator(
    session: Session,
    organization: Organization,
    user: User,
    epoch_assignment: RoleAssignment,
    *,
    branch: Branch,
    assessment: RiskAssessment,
    admission_parent: ReferenceProfile,
    active_personal_profile: ReferenceProfile | None,
) -> PersonalReferenceAccumulator:
    _lock_personal_accumulator_scope(
        session,
        organization,
        user,
        epoch_assignment,
        branch=branch,
        model_version=assessment.model_version,
        config_version=assessment.config_version,
    )
    accumulator = _personal_accumulator_for_scope(
        session,
        organization,
        user,
        epoch_assignment,
        branch=branch,
        model_version=assessment.model_version,
        config_version=assessment.config_version,
        lock=True,
    )
    if accumulator is not None:
        return accumulator
    accumulator = PersonalReferenceAccumulator(
        organization_id=organization.id,
        user_id=user.id,
        role_assignment_id=epoch_assignment.id,
        branch=branch,
        model_version=assessment.model_version,
        config_version=assessment.config_version,
        catalog_version=_catalog_version_for_branch(branch),
        admission_reference_profile_id=admission_parent.id,
        active_reference_profile_id=(
            active_personal_profile.id if active_personal_profile is not None else None
        ),
        status=(
            PersonalAccumulatorStatus.ACTIVE
            if active_personal_profile is not None
            else PersonalAccumulatorStatus.WARMING
        ),
        release_sequence=(
            active_personal_profile.reference_version or 1
            if active_personal_profile is not None
            else 0
        ),
        last_release_day=(
            active_personal_profile.release_day
            if active_personal_profile is not None
            else None
        ),
    )
    session.add(accumulator)
    session.flush()
    return accumulator


def _record_safe_update_admission(
    session: Session,
    organization: Organization,
    user: User,
    epoch_assignment: RoleAssignment,
    *,
    assessment: RiskAssessment,
    branch_score: BranchScore,
    branch_input: BranchAssessmentInput,
    profiles: dict[ReferenceLevel, ReferenceProfile],
    role_known: bool,
    framework_config: Mapping[str, Any],
    policy: SafeUpdatePolicy,
    actor: str,
    request_id: str,
) -> SafeUpdateCandidate | None:
    reasons: list[str] = []
    if assessment.split is not DataSplit.PRODUCTION:
        reasons.append("PRIMARY_EXPERIMENT_UPDATE_FORBIDDEN")
    if assessment.status is not AssessmentStatus.SCORED:
        reasons.append("SOURCE_ASSESSMENT_NOT_SCORED")
    if assessment.is_alert:
        reasons.append("DAY_IS_ALERT")
    if branch_score.status is not ScoreStatus.SCORED or branch_score.raw_score is None:
        reasons.append("SOURCE_BRANCH_NOT_SCORED")

    existing_accumulator = _personal_accumulator_for_scope(
        session,
        organization,
        user,
        epoch_assignment,
        branch=branch_score.branch,
        model_version=assessment.model_version,
        config_version=assessment.config_version,
    )
    if existing_accumulator is not None:
        parent = session.get(
            ReferenceProfile,
            existing_accumulator.admission_reference_profile_id,
        )
        if (
            parent is None
            or parent.organization_id != organization.id
            or parent.branch is not branch_score.branch
            or parent.level not in {ReferenceLevel.ROLE, ReferenceLevel.GLOBAL}
            or not parent.is_frozen
        ):
            reasons.append("PINNED_ADMISSION_PARENT_INVALID")
            parent = None
    else:
        parent, parent_reasons = _admission_parent_for_branch(
            branch=branch_score.branch,
            score_date=assessment.day,
            branch_input=branch_input,
            profiles=profiles,
            role_known=role_known,
            framework_config=framework_config,
        )
        reasons.extend(parent_reasons)
    observation, contribution = _candidate_source_observation(
        session,
        organization,
        user,
        branch=branch_score.branch,
        candidate_day=assessment.day,
    )
    if observation is None:
        reasons.append("SOURCE_USER_DAY_MISSING_OR_INACTIVE")
    elif isinstance(observation, UserDaySequence):
        if branch_input.seq_len != observation.seq_len:
            reasons.append("SOURCE_SEQUENCE_LENGTH_MISMATCH")
        if (
            branch_input.truncated is not None
            and branch_input.truncated != observation.truncated
        ):
            reasons.append("SOURCE_SEQUENCE_TRUNCATION_MISMATCH")

    percentile: float | None = None
    if parent is not None and branch_score.raw_score is not None:
        try:
            percentile = empirical_percentile(
                float(branch_score.raw_score),
                parent.calibrator_json["sorted_scores"],
            )
        except (KeyError, TypeError, ValueError):
            reasons.append("ADMISSION_PARENT_CALIBRATOR_INVALID")
        else:
            if not admission_allowed(
                percentile,
                policy.admission_max_percentile_exclusive,
            ):
                reasons.append("PARENT_PERCENTILE_NOT_BELOW_THRESHOLD")

    if reasons:
        append_audit(
            session,
            organization,
            actor=actor,
            action="safe_update.admission_rejected",
            entity_type="branch_score",
            entity_id=str(branch_score.id),
            request_id=request_id,
            after={
                "branch": branch_score.branch.value,
                "day": assessment.day,
                "user_id": user.external_user_id,
                "parent_reference_profile_id": str(parent.id) if parent else None,
                "parent_percentile": percentile,
                "threshold_exclusive": policy.admission_max_percentile_exclusive,
                "reason_codes": sorted(set(reasons)),
                "policy_version": policy.policy_version,
            },
        )
        return None

    assert parent is not None
    assert observation is not None
    assert percentile is not None
    active_personal = profiles.get(ReferenceLevel.PERSON)
    accumulator = _get_or_create_personal_accumulator(
        session,
        organization,
        user,
        epoch_assignment,
        branch=branch_score.branch,
        assessment=assessment,
        admission_parent=parent,
        active_personal_profile=active_personal,
    )
    quarantine_until, eligible_on = quarantine_window(assessment.day, policy)
    candidate = SafeUpdateCandidate(
        organization_id=organization.id,
        user_id=user.id,
        role_assignment_id=epoch_assignment.id,
        reference_profile_id=accumulator.active_reference_profile_id,
        source_assessment_id=assessment.id,
        accumulator_id=accumulator.id,
        source_branch_score_id=branch_score.id,
        source_feature_id=(
            observation.id if isinstance(observation, UserDayFeature) else None
        ),
        source_sequence_id=(
            observation.id if isinstance(observation, UserDaySequence) else None
        ),
        source_input_checksum=observation.input_checksum,
        admission_reference_profile_id=parent.id,
        admission_reference_checksum=parent.checksum,
        admission_percentile=percentile,
        admission_threshold=policy.admission_max_percentile_exclusive,
        branch=branch_score.branch,
        candidate_day=assessment.day,
        quarantine_until=quarantine_until,
        eligible_on=eligible_on,
        status=UpdateStatus.CANDIDATE,
        reason_codes=["ADMISSION_PASSED"],
        support_contribution_json=contribution,
        policy_version=policy.policy_version,
        model_version=assessment.model_version,
        config_version=assessment.config_version,
        influence_cap=policy.per_release_influence_cap,
    )
    session.add(candidate)
    session.flush()
    append_audit(
        session,
        organization,
        actor=actor,
        action="safe_update.candidate_created",
        entity_type="safe_update_candidate",
        entity_id=str(candidate.id),
        request_id=request_id,
        after={
            "branch": candidate.branch.value,
            "candidate_day": candidate.candidate_day,
            "eligible_on": candidate.eligible_on,
            "parent_reference_profile_id": str(parent.id),
            "parent_percentile": percentile,
            "threshold_exclusive": policy.admission_max_percentile_exclusive,
            "policy_version": policy.policy_version,
        },
    )
    return candidate


def _production_person_staleness_gap(
    session: Session,
    user: User,
    assignment: RoleAssignment,
    branch: Branch,
    score_date: date,
    profile: ReferenceProfile,
) -> int:
    epoch_assignment = role_epoch_anchor(session, assignment)
    if branch is Branch.FEATURE:
        latest_active_day = session.scalar(
            select(func.max(UserDayFeature.day))
            .join(FeatureCatalog, FeatureCatalog.id == UserDayFeature.catalog_id)
            .where(
                UserDayFeature.user_id == user.id,
                UserDayFeature.day >= epoch_assignment.valid_from,
                UserDayFeature.day < score_date,
                UserDayFeature.is_active_day.is_(True),
                FeatureCatalog.version == FEATURE_SCHEMA_VERSION,
            )
        )
    else:
        latest_active_day = session.scalar(
            select(func.max(UserDaySequence.day)).where(
                UserDaySequence.user_id == user.id,
                UserDaySequence.day >= epoch_assignment.valid_from,
                UserDaySequence.day < score_date,
                UserDaySequence.vocabulary_version == SEQUENCE_SCHEMA_VERSION,
                UserDaySequence.seq_len >= 2,
            )
        )
    stored_last_active = (
        profile.support_json.get("last_active_day")
        if branch is Branch.FEATURE
        else profile.support_json.get("last_sequence_day")
        or profile.support_json.get("last_active_day")
    )
    stored_day = (
        date.fromisoformat(str(stored_last_active))
        if stored_last_active
        else profile.fitted_through
    )
    latest = max(
        day_value
        for day_value in (latest_active_day, stored_day)
        if day_value is not None
    )
    return max((score_date - latest).days, 0)


def create_assessment(
    session: Session,
    organization: Organization,
    payload: AssessmentCreate,
    *,
    actor: str,
    request_id: str,
    locked_threshold: float | None,
) -> RiskAssessment:
    if payload.feature is not None:
        feature_context = {"evidence": payload.feature.evidence}
        enforce_label_firewall(feature_context)
        enforce_metadata_only_payload(feature_context)
        top_features = payload.feature.evidence.get("top_features", [])
        if top_features:
            allowed_features = {
                str(item["name"])
                for item in load_feature_catalog().get("features", [])
                if isinstance(item, dict) and item.get("name")
            }
            unknown_features = sorted(set(top_features) - allowed_features)
            if unknown_features:
                raise DomainValidationError(
                    "FEATURE_EVIDENCE_UNKNOWN",
                    "feature evidence contains names outside the active catalog",
                    {"features": unknown_features},
                )
    if payload.sequence is not None:
        sequence_context = {"evidence": payload.sequence.evidence}
        enforce_label_firewall(sequence_context)
        enforce_metadata_only_payload(sequence_context)
    expected_split = dataset_split(payload.day)
    if payload.split != expected_split:
        raise DomainValidationError(
            "SPLIT_MISMATCH",
            "split must be derived from the locked date boundaries",
            {"expected": expected_split, "provided": payload.split},
        )

    framework_config = load_framework_config()
    configured_version = str(framework_config["schema_version"])
    if payload.config_version != configured_version:
        raise DomainValidationError(
            "CONFIG_VERSION_NOT_ACTIVE",
            "assessment config_version must match the active immutable framework config",
            {
                "active": configured_version,
                "provided": payload.config_version,
            },
        )
    configured_feature_weight = float(framework_config["fusion"]["weights"]["feature"])
    configured_sequence_weight = float(framework_config["fusion"]["weights"]["sequence"])
    configured_threshold_value = framework_config.get("decision", {}).get("alert_threshold")
    if configured_threshold_value is not None and locked_threshold is not None:
        if not math.isclose(float(configured_threshold_value), locked_threshold):
            raise DomainValidationError(
                "THRESHOLD_RELEASE_CONFLICT",
                "framework config and runtime threshold release do not match",
                {
                    "framework": configured_threshold_value,
                    "runtime": locked_threshold,
                },
            )
    configured_threshold = (
        float(configured_threshold_value)
        if configured_threshold_value is not None
        else locked_threshold
    )
    if configured_threshold is None:
        raise DomainValidationError(
            "THRESHOLD_NOT_LOCKED",
            "score persistence requires LOCKED_ALERT_THRESHOLD from the Validation release",
        )
    if not (
        math.isclose(payload.feature_weight, configured_feature_weight)
        and math.isclose(payload.sequence_weight, configured_sequence_weight)
    ):
        raise DomainValidationError(
            "FUSION_CONFIG_MISMATCH",
            "fusion weights are controlled by the active framework config",
            {
                "feature_weight": configured_feature_weight,
                "sequence_weight": configured_sequence_weight,
            },
        )
    if payload.alert_threshold is not None and not math.isclose(
        payload.alert_threshold,
        configured_threshold,
    ):
        raise DomainValidationError(
            "ALERT_THRESHOLD_MISMATCH",
            "alert threshold is controlled by server configuration",
            {"configured": configured_threshold},
        )

    request_snapshot = payload.model_dump(mode="json")
    request_snapshot.update(
        {
            "feature_weight": configured_feature_weight,
            "sequence_weight": configured_sequence_weight,
            "alert_threshold": configured_threshold,
        }
    )
    request_hash = sha256_json(request_snapshot)

    user = get_user(session, organization, payload.user_id)
    _lock_assessment_release(
        session,
        organization_id=organization.id,
        user_id=user.id,
        day=payload.day,
        model_version=payload.model_version,
        config_version=payload.config_version,
    )
    existing = session.scalar(
        select(RiskAssessment).where(
            RiskAssessment.organization_id == organization.id,
            RiskAssessment.user_id == user.id,
            RiskAssessment.day == payload.day,
            RiskAssessment.model_version == payload.model_version,
            RiskAssessment.config_version == payload.config_version,
        )
    )
    if existing is not None:
        persisted_hash = existing.fusion_evidence.get("request_hash")
        if persisted_hash is not None and persisted_hash != request_hash:
            raise conflict(
                "ASSESSMENT_RELEASE_CONFLICT",
                "the model/config release already exists with different scoring inputs",
                {
                    "user_id": payload.user_id,
                    "day": payload.day.isoformat(),
                    "model_version": payload.model_version,
                    "config_version": payload.config_version,
                },
            )
        return existing

    closed_watermark = session.scalar(
        select(ScoringWatermark.id).where(
            ScoringWatermark.organization_id == organization.id,
            ScoringWatermark.day == payload.day,
            ScoringWatermark.model_version == payload.model_version,
            ScoringWatermark.config_version == payload.config_version,
        )
    )
    if closed_watermark is not None:
        raise conflict(
            "SCORING_DAY_ALREADY_CLOSED",
            "the immutable scoring watermark forbids a new assessment release for this day",
            {
                "day": payload.day.isoformat(),
                "model_version": payload.model_version,
                "config_version": payload.config_version,
                "watermark_id": str(closed_watermark),
            },
        )

    assignment = role_assignment_at(session, user.id, payload.day)
    role_known = bool(assignment and not assignment.role.is_unknown)
    scoring_run_id = request_id
    feature_score = None
    sequence_score = None
    profiles_by_branch: dict[Branch, dict[ReferenceLevel, ReferenceProfile]] = {}

    if payload.feature is not None:
        feature_profiles = _profiles_for_branch(
            session,
            organization,
            user,
            assignment,
            branch=Branch.FEATURE,
            assessment=payload,
            branch_input=payload.feature,
        )
        profiles_by_branch[Branch.FEATURE] = feature_profiles
        supports = _build_feature_supports(feature_profiles, role_known)
        if (
            payload.split == "PRODUCTION"
            and assignment is not None
            and supports[0] is not None
            and ReferenceLevel.PERSON in feature_profiles
        ):
            supports = (
                replace(
                    supports[0],
                    last_active_gap_days=_production_person_staleness_gap(
                        session,
                        user,
                        assignment,
                        Branch.FEATURE,
                        payload.day,
                        feature_profiles[ReferenceLevel.PERSON],
                    ),
                ),
                supports[1],
                supports[2],
            )
        feature_decision = select_feature_reference(
            score_date=payload.day,
            person=supports[0],
            role=supports[1],
            global_support=supports[2],
            config=framework_config,
        )
        feature_score = _persist_branch_score(
            session,
            organization,
            user,
            assignment,
            branch=Branch.FEATURE,
            payload=payload.feature,
            decision=feature_decision,
            profiles=feature_profiles,
            assessment=payload,
            scoring_run_id=scoring_run_id,
        )
    if payload.sequence is not None:
        if payload.sequence.seq_len is None:
            raise DomainValidationError(
                "SEQUENCE_LENGTH_REQUIRED",
                "sequence branch assessment requires seq_len",
            )
        sequence_profiles = _profiles_for_branch(
            session,
            organization,
            user,
            assignment,
            branch=Branch.SEQUENCE,
            assessment=payload,
            branch_input=payload.sequence,
        )
        profiles_by_branch[Branch.SEQUENCE] = sequence_profiles
        supports = _build_sequence_supports(sequence_profiles, role_known)
        if (
            payload.split == "PRODUCTION"
            and assignment is not None
            and supports[0] is not None
            and ReferenceLevel.PERSON in sequence_profiles
        ):
            supports = (
                replace(
                    supports[0],
                    last_active_gap_days=_production_person_staleness_gap(
                        session,
                        user,
                        assignment,
                        Branch.SEQUENCE,
                        payload.day,
                        sequence_profiles[ReferenceLevel.PERSON],
                    ),
                ),
                supports[1],
                supports[2],
            )
        sequence_decision = select_sequence_reference(
            score_date=payload.day,
            seq_len=payload.sequence.seq_len,
            person=supports[0],
            role=supports[1],
            global_support=supports[2],
            config=framework_config,
        )
        sequence_score = _persist_branch_score(
            session,
            organization,
            user,
            assignment,
            branch=Branch.SEQUENCE,
            payload=payload.sequence,
            decision=sequence_decision,
            profiles=sequence_profiles,
            assessment=payload,
            scoring_run_id=scoring_run_id,
        )

    q_feature = (
        feature_score.calibrated_score
        if feature_score and feature_score.status is ScoreStatus.SCORED
        else None
    )
    q_sequence = (
        sequence_score.calibrated_score
        if sequence_score and sequence_score.status is ScoreStatus.SCORED
        else None
    )
    fusion = fuse_scores(
        q_feature=q_feature,
        q_sequence=q_sequence,
        config={
            "version": payload.config_version,
            "fusion": {
                "feature_weight": configured_feature_weight,
                "sequence_weight": configured_sequence_weight,
            },
        },
    )
    threshold = configured_threshold
    is_alert = bool(fusion.risk is not None and fusion.risk >= threshold)
    assessment = RiskAssessment(
        organization_id=organization.id,
        user_id=user.id,
        day=payload.day,
        role_assignment_id=assignment.id if assignment else None,
        split=DataSplit(payload.split.lower()),
        status=(AssessmentStatus.SCORED if fusion.risk is not None else AssessmentStatus.NO_SCORE),
        feature_score_id=(
            feature_score.id
            if feature_score and feature_score.status is ScoreStatus.SCORED
            else None
        ),
        sequence_score_id=(
            sequence_score.id
            if sequence_score and sequence_score.status is ScoreStatus.SCORED
            else None
        ),
        feature_weight=fusion.feature_weight,
        sequence_weight=fusion.sequence_weight,
        risk=fusion.risk,
        threshold=threshold if fusion.risk is not None else None,
        is_alert=is_alert,
        model_version=payload.model_version,
        config_version=payload.config_version,
        scoring_run_id=scoring_run_id,
        fusion_evidence={
            "available_branches": list(fusion.available_branches),
            "reason_codes": list(fusion.reason_codes),
            "q_feature": q_feature,
            "q_sequence": q_sequence,
            "request_hash": request_hash,
        },
    )
    session.add(assessment)
    session.flush()

    if is_alert:
        session.add(
            Alert(
                organization_id=organization.id,
                assessment_id=assessment.id,
                severity=AlertSeverity(risk_severity(fusion.risk).lower()),
            )
        )

    safe_update_config = framework_config["safe_personalized_update"]
    safe_update_policy_version = safe_update_config.get("policy_version")
    safe_update_enabled = (
        payload.split == "PRODUCTION"
        and bool(safe_update_config.get("production_enabled", True))
        and payload.config_version == str(framework_config["schema_version"])
        and isinstance(safe_update_policy_version, str)
        and safe_update_policy_version.startswith("framework.v5.")
    )
    if assignment is not None and safe_update_enabled:
        epoch_assignment = role_epoch_anchor(session, assignment)
        policy = SafeUpdatePolicy.from_framework(framework_config)
        for branch, branch_input, branch_score in (
            (Branch.FEATURE, payload.feature, feature_score),
            (Branch.SEQUENCE, payload.sequence, sequence_score),
        ):
            if branch_input is None or branch_score is None:
                continue
            _record_safe_update_admission(
                session,
                organization,
                user,
                epoch_assignment,
                assessment=assessment,
                branch_score=branch_score,
                branch_input=branch_input,
                profiles=profiles_by_branch[branch],
                role_known=role_known,
                framework_config=framework_config,
                policy=policy,
                actor=actor,
                request_id=request_id,
            )
    session.flush()
    append_audit(
        session,
        organization,
        actor=actor,
        action="risk_assessment.created",
        entity_type="risk_assessment",
        entity_id=str(assessment.id),
        request_id=request_id,
        after={
            "user_id": user.external_user_id,
            "day": payload.day,
            "risk": assessment.risk,
            "is_alert": assessment.is_alert,
        },
    )
    return assessment


def get_assessment(
    session: Session,
    organization: Organization,
    external_user_id: str,
    day: date,
    model_version: str,
    config_version: str,
) -> tuple[
    RiskAssessment,
    User,
    Role | None,
    BranchScore | None,
    BranchScore | None,
    Alert | None,
    list[SafeUpdateCandidate],
]:
    user = get_user(session, organization, external_user_id)
    assessment = session.scalar(
        select(RiskAssessment).where(
            RiskAssessment.organization_id == organization.id,
            RiskAssessment.user_id == user.id,
            RiskAssessment.day == day,
            RiskAssessment.model_version == model_version,
            RiskAssessment.config_version == config_version,
        )
    )
    if assessment is None:
        raise not_found(
            "risk assessment",
            f"{external_user_id}/{day}/{model_version}/{config_version}",
        )
    assignment = (
        session.get(RoleAssignment, assessment.role_assignment_id)
        if assessment.role_assignment_id
        else None
    )
    role = session.get(Role, assignment.role_id) if assignment else None
    feature = (
        session.get(BranchScore, assessment.feature_score_id)
        if assessment.feature_score_id
        else session.scalar(
            select(BranchScore).where(
                BranchScore.user_id == user.id,
                BranchScore.day == day,
                BranchScore.branch == Branch.FEATURE,
                BranchScore.model_version == model_version,
                BranchScore.config_version == assessment.config_version,
            )
        )
    )
    sequence = (
        session.get(BranchScore, assessment.sequence_score_id)
        if assessment.sequence_score_id
        else session.scalar(
            select(BranchScore).where(
                BranchScore.user_id == user.id,
                BranchScore.day == day,
                BranchScore.branch == Branch.SEQUENCE,
                BranchScore.model_version == model_version,
                BranchScore.config_version == assessment.config_version,
            )
        )
    )
    alert = session.scalar(select(Alert).where(Alert.assessment_id == assessment.id))
    candidates = list(
        session.scalars(
            select(SafeUpdateCandidate).where(
                SafeUpdateCandidate.source_assessment_id == assessment.id
            )
        )
    )
    return assessment, user, role, feature, sequence, alert, candidates


def list_alerts(
    session: Session,
    organization: Organization,
    *,
    status: str | None,
    limit: int,
    offset: int,
) -> list[tuple[Alert, RiskAssessment, User]]:
    statement = (
        select(Alert, RiskAssessment, User)
        .join(RiskAssessment, RiskAssessment.id == Alert.assessment_id)
        .join(User, User.id == RiskAssessment.user_id)
        .where(Alert.organization_id == organization.id)
    )
    if status:
        statement = statement.where(Alert.status == AlertStatus(status.lower()))
    return list(
        session.execute(
            statement.order_by(Alert.opened_at.desc()).offset(offset).limit(limit)
        ).all()
    )


def patch_alert(
    session: Session,
    organization: Organization,
    alert_id: str,
    payload: AlertPatch,
    *,
    actor: str,
    request_id: str,
) -> tuple[Alert, RiskAssessment, User]:
    try:
        parsed = uuid.UUID(alert_id)
    except ValueError as exc:
        raise not_found("alert", alert_id) from exc
    row = session.execute(
        select(Alert, RiskAssessment, User)
        .join(RiskAssessment, RiskAssessment.id == Alert.assessment_id)
        .join(User, User.id == RiskAssessment.user_id)
        .where(Alert.id == parsed, Alert.organization_id == organization.id)
    ).one_or_none()
    if row is None:
        raise not_found("alert", alert_id)
    alert, assessment, user = row
    before = {
        "status": alert.status.value,
        "assignee": alert.assignee,
        "resolution": alert.resolution,
    }
    if payload.status is not None:
        alert.status = AlertStatus(payload.status.lower())
    if "assignee" in payload.model_fields_set:
        alert.assignee = payload.assignee
    if "resolution" in payload.model_fields_set:
        alert.resolution = payload.resolution
    if alert.status in {
        AlertStatus.CLOSED,
        AlertStatus.RESOLVED,
        AlertStatus.DISMISSED,
        AlertStatus.FALSE_POSITIVE,
    }:
        alert.closed_at = utc_now()
    else:
        alert.closed_at = None
    session.flush()
    append_audit(
        session,
        organization,
        actor=actor,
        action="alert.updated",
        entity_type="alert",
        entity_id=str(alert.id),
        request_id=request_id,
        before=before,
        after={
            "status": alert.status.value,
            "assignee": alert.assignee,
            "resolution": alert.resolution,
        },
    )
    return alert, assessment, user


def close_scoring_day(
    session: Session,
    organization: Organization,
    *,
    day: date,
    model_version: str,
    config_version: str,
    actor: str,
    request_id: str,
) -> ScoringWatermark:
    framework_config = load_framework_config()
    active_config_version = str(framework_config["schema_version"])
    if config_version != active_config_version:
        raise DomainValidationError(
            "WATERMARK_CONFIG_NOT_ACTIVE",
            "a scoring watermark can only close the active framework release",
            {"active": active_config_version, "provided": config_version},
        )
    if dataset_split(day) != "PRODUCTION":
        raise DomainValidationError(
            "WATERMARK_PRODUCTION_ONLY",
            "safe-update scoring watermarks are production-only",
            {"day": day.isoformat()},
        )

    universe_ids = sorted(
        {
            str(user_id)
            for user_id in session.scalars(
                select(RoleAssignment.user_id)
                .join(User, User.id == RoleAssignment.user_id)
                .where(
                    User.organization_id == organization.id,
                    RoleAssignment.valid_from <= day,
                    or_(RoleAssignment.valid_to.is_(None), RoleAssignment.valid_to > day),
                )
            )
        }
    )
    assessments = list(
        session.scalars(
            select(RiskAssessment).where(
                RiskAssessment.organization_id == organization.id,
                RiskAssessment.day == day,
                RiskAssessment.model_version == model_version,
                RiskAssessment.config_version == config_version,
            )
        )
    )
    assessment_user_ids = {str(item.user_id) for item in assessments}
    universe_id_set = set(universe_ids)
    if assessment_user_ids != universe_id_set:
        raise DomainValidationError(
            "SCORING_DAY_INCOMPLETE",
            "the persisted assessment set does not exactly match the effective-dated role universe",
            {
                "day": day.isoformat(),
                "expected_assessments": len(universe_ids),
                "persisted_assessments": len(assessments),
                "missing_user_ids": sorted(universe_id_set - assessment_user_ids),
                "unexpected_user_ids": sorted(assessment_user_ids - universe_id_set),
            },
        )
    if any(item.split is not DataSplit.PRODUCTION for item in assessments):
        raise DomainValidationError(
            "WATERMARK_NON_PRODUCTION_ASSESSMENT",
            "the watermark set contains a non-production assessment",
        )

    universe_checksum = sha256_json(universe_ids)
    assessment_set_checksum = sha256_json(
        [
            {
                "assessment_id": str(item.id),
                "user_id": str(item.user_id),
                "status": item.status.value,
                "is_alert": item.is_alert,
            }
            for item in sorted(assessments, key=lambda value: str(value.user_id))
        ]
    )
    existing = session.scalar(
        select(ScoringWatermark).where(
            ScoringWatermark.organization_id == organization.id,
            ScoringWatermark.day == day,
            ScoringWatermark.model_version == model_version,
            ScoringWatermark.config_version == config_version,
        )
    )
    if existing is not None:
        if (
            existing.universe_checksum != universe_checksum
            or existing.assessment_set_checksum != assessment_set_checksum
            or existing.expected_assessments != len(universe_ids)
            or existing.persisted_assessments != len(assessments)
        ):
            raise conflict(
                "WATERMARK_RELEASE_CONFLICT",
                "the immutable scoring watermark already exists with different evidence",
                {"watermark_id": str(existing.id), "day": day.isoformat()},
            )
        return existing

    watermark = ScoringWatermark(
        organization_id=organization.id,
        day=day,
        model_version=model_version,
        config_version=config_version,
        expected_assessments=len(universe_ids),
        persisted_assessments=len(assessments),
        universe_checksum=universe_checksum,
        assessment_set_checksum=assessment_set_checksum,
        completed_at=utc_now(),
    )
    session.add(watermark)
    session.flush()
    append_audit(
        session,
        organization,
        actor=actor,
        action="scoring_watermark.closed",
        entity_type="scoring_watermark",
        entity_id=str(watermark.id),
        request_id=request_id,
        after={
            "day": day,
            "model_version": model_version,
            "config_version": config_version,
            "expected_assessments": len(universe_ids),
            "assessment_set_checksum": assessment_set_checksum,
        },
    )
    return watermark


def _quarantine_watermarks_complete(
    session: Session,
    organization: Organization,
    candidate: SafeUpdateCandidate,
    policy: SafeUpdatePolicy,
) -> bool:
    watermark_count = session.scalar(
        select(func.count(ScoringWatermark.id)).where(
            ScoringWatermark.organization_id == organization.id,
            ScoringWatermark.model_version == candidate.model_version,
            ScoringWatermark.config_version == candidate.config_version,
            ScoringWatermark.day >= candidate.candidate_day,
            ScoringWatermark.day <= candidate.quarantine_until,
        )
    )
    return int(watermark_count or 0) == policy.quarantine_days + 1


def _candidate_source_is_unchanged(
    session: Session,
    candidate: SafeUpdateCandidate,
) -> bool:
    source: UserDayFeature | UserDaySequence | None
    if candidate.branch is Branch.FEATURE:
        source = (
            session.get(UserDayFeature, candidate.source_feature_id)
            if candidate.source_feature_id
            else None
        )
    else:
        source = (
            session.get(UserDaySequence, candidate.source_sequence_id)
            if candidate.source_sequence_id
            else None
        )
    return bool(
        source is not None
        and source.user_id == candidate.user_id
        and source.day == candidate.candidate_day
        and source.input_checksum == candidate.source_input_checksum
    )


def _candidate_integrity_reasons(
    session: Session,
    organization: Organization,
    candidate: SafeUpdateCandidate,
    *,
    latest_watermark_day: date,
    policy: SafeUpdatePolicy,
) -> list[str]:
    reasons: list[str] = []
    assessment = session.get(RiskAssessment, candidate.source_assessment_id)
    branch_score = (
        session.get(BranchScore, candidate.source_branch_score_id)
        if candidate.source_branch_score_id
        else None
    )
    parent = (
        session.get(ReferenceProfile, candidate.admission_reference_profile_id)
        if candidate.admission_reference_profile_id
        else None
    )
    if assessment is None:
        reasons.append("SOURCE_ASSESSMENT_MISSING")
    else:
        if assessment.organization_id != organization.id:
            reasons.append("SOURCE_ORGANIZATION_MISMATCH")
        if assessment.split is not DataSplit.PRODUCTION:
            reasons.append("PRIMARY_EXPERIMENT_UPDATE_FORBIDDEN")
        if assessment.status is not AssessmentStatus.SCORED:
            reasons.append("SOURCE_ASSESSMENT_NOT_SCORED")
        if assessment.is_alert:
            reasons.append("DAY_IS_ALERT")
        if (
            assessment.model_version != candidate.model_version
            or assessment.config_version != candidate.config_version
        ):
            reasons.append("SOURCE_RELEASE_MISMATCH")
    if (
        branch_score is None
        or branch_score.status is not ScoreStatus.SCORED
        or branch_score.raw_score is None
        or branch_score.user_id != candidate.user_id
        or branch_score.day != candidate.candidate_day
        or branch_score.branch is not candidate.branch
    ):
        reasons.append("SOURCE_BRANCH_SCORE_INVALID")
    if not _candidate_source_is_unchanged(session, candidate):
        reasons.append("SOURCE_USER_DAY_CHANGED")
    if (
        parent is None
        or parent.level not in {ReferenceLevel.ROLE, ReferenceLevel.GLOBAL}
        or parent.branch is not candidate.branch
        or not parent.is_frozen
        or parent.checksum != candidate.admission_reference_checksum
    ):
        reasons.append("ADMISSION_PARENT_CHANGED_OR_INVALID")
    elif branch_score is not None and branch_score.raw_score is not None:
        try:
            recalculated = empirical_percentile(
                float(branch_score.raw_score),
                parent.calibrator_json["sorted_scores"],
            )
        except (KeyError, TypeError, ValueError):
            reasons.append("ADMISSION_PARENT_CALIBRATOR_INVALID")
        else:
            if (
                candidate.admission_percentile is None
                or not math.isclose(
                    recalculated,
                    candidate.admission_percentile,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or candidate.admission_threshold is None
                or not math.isclose(
                    candidate.admission_threshold,
                    policy.admission_max_percentile_exclusive,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not admission_allowed(
                    recalculated,
                    policy.admission_max_percentile_exclusive,
                )
            ):
                reasons.append("ADMISSION_SNAPSHOT_INVALID")

    quarantine_alert = session.scalar(
        select(RiskAssessment.id).where(
            RiskAssessment.organization_id == organization.id,
            RiskAssessment.user_id == candidate.user_id,
            RiskAssessment.is_alert.is_(True),
            RiskAssessment.day >= candidate.candidate_day,
            RiskAssessment.day <= candidate.quarantine_until,
        )
    )
    if quarantine_alert is not None:
        reasons.append("ALERT_IN_QUARANTINE_WINDOW")

    for checkpoint_day in (candidate.quarantine_until, latest_watermark_day):
        assignment = role_assignment_at(session, candidate.user_id, checkpoint_day)
        epoch = role_epoch_anchor(session, assignment) if assignment is not None else None
        if epoch is None or epoch.id != candidate.role_assignment_id:
            reasons.append("ROLE_EPOCH_CHANGED")
            break
    return sorted(set(reasons))


def _candidate_raw_scores(
    session: Session,
    candidates: Sequence[SafeUpdateCandidate],
) -> list[float]:
    scores: list[float] = []
    for candidate in candidates:
        score = session.get(BranchScore, candidate.source_branch_score_id)
        if score is None or score.raw_score is None or not math.isfinite(score.raw_score):
            raise RuntimeError("accepted safe-update candidate lost its immutable raw score")
        scores.append(float(score.raw_score))
    return scores


def _feature_candidate_support(
    candidates: Sequence[SafeUpdateCandidate],
) -> tuple[FeatureBootstrapSupport, list[int]]:
    if not candidates:
        return FeatureBootstrapSupport(0, 0, 0, 0.0, 0), []
    dimension = int(candidates[0].support_contribution_json.get("feature_dimension", 0))
    if dimension < 1:
        return FeatureBootstrapSupport(0, 0, 0, 0.0, 0), []
    counts = [0] * dimension
    for candidate in candidates:
        contribution = candidate.support_contribution_json
        if int(contribution.get("feature_dimension", 0)) != dimension:
            return FeatureBootstrapSupport(0, 0, 0, 0.0, 0), []
        ordinals = contribution.get("present_ordinals", [])
        if not isinstance(ordinals, list):
            return FeatureBootstrapSupport(0, 0, 0, 0.0, 0), []
        for ordinal in ordinals:
            index = int(ordinal) - 1
            if index < 0 or index >= dimension:
                return FeatureBootstrapSupport(0, 0, 0, 0.0, 0), []
            counts[index] += 1
    active_days = len(candidates)
    span_days = (candidates[-1].candidate_day - candidates[0].candidate_day).days + 1
    used_counts = [value for value in counts if value > 0]
    coverage = sum(counts) / (active_days * dimension)
    support = FeatureBootstrapSupport(
        active_days=active_days,
        span_days=span_days,
        active_days_current_role=active_days,
        coverage=coverage,
        min_feature_observations=min(used_counts) if used_counts else 0,
    )
    return support, counts


def _sequence_candidate_support(
    candidates: Sequence[SafeUpdateCandidate],
) -> SequenceBootstrapSupport:
    if not candidates:
        return SequenceBootstrapSupport(0, 0, 0, 0)
    transitions = sum(
        int(candidate.support_contribution_json.get("transitions", 0))
        for candidate in candidates
    )
    return SequenceBootstrapSupport(
        sequence_days=len(candidates),
        span_days=(candidates[-1].candidate_day - candidates[0].candidate_day).days + 1,
        sequence_days_current_role=len(candidates),
        transitions=transitions,
    )


def _bootstrap_candidate_prefix(
    candidates: Sequence[SafeUpdateCandidate],
    *,
    branch: Branch,
    framework_config: Mapping[str, Any],
) -> list[SafeUpdateCandidate]:
    selected: list[SafeUpdateCandidate] = []
    for candidate in candidates:
        selected.append(candidate)
        if branch is Branch.FEATURE:
            support, _ = _feature_candidate_support(selected)
            decision = feature_bootstrap_readiness(support, framework_config)
        else:
            support = _sequence_candidate_support(selected)
            decision = sequence_bootstrap_readiness(support, framework_config)
        if decision.ready:
            return selected
    return []


def _profile_personal_scores(profile: ReferenceProfile) -> list[float]:
    method = profile.calibrator_json.get("method")
    if method == "empirical_cdf.v1":
        values = profile.calibrator_json.get("sorted_scores")
    elif method == "parent_shrunk_ecdf.v1":
        values = profile.calibrator_json.get("personal_sorted_scores")
    else:
        raise RuntimeError("active Personal reference has an unsupported calibrator")
    if not isinstance(values, list) or not values:
        raise RuntimeError("active Personal reference has no calibration scores")
    normalized = [float(value) for value in values]
    if any(not math.isfinite(value) for value in normalized):
        raise RuntimeError("active Personal reference contains non-finite scores")
    return normalized


def _child_support_snapshot(
    *,
    accumulator: PersonalReferenceAccumulator,
    parent: ReferenceProfile | None,
    candidates: Sequence[SafeUpdateCandidate],
    release_day: date,
) -> tuple[dict[str, Any], int, int, float | None, date]:
    first_day = candidates[0].candidate_day
    last_day = candidates[-1].candidate_day
    parent_first_raw = None
    if parent is not None:
        parent_first_raw = (
            parent.support_json.get("first_active_day")
            if accumulator.branch is Branch.FEATURE
            else parent.support_json.get("first_sequence_day")
            or parent.support_json.get("first_active_day")
        )
    parent_first = (
        date.fromisoformat(str(parent_first_raw))
        if parent_first_raw
        else (parent.fitted_from if parent and parent.fitted_from else None)
    )
    if parent is not None and parent_first is None:
        parent_last_raw = (
            parent.support_json.get("last_active_day")
            if accumulator.branch is Branch.FEATURE
            else parent.support_json.get("last_sequence_day")
            or parent.support_json.get("last_active_day")
        )
        parent_last = (
            date.fromisoformat(str(parent_last_raw))
            if parent_last_raw
            else parent.fitted_through
        )
        parent_span_days = max(int(parent.support_json.get("span_days", 1)), 1)
        parent_first = parent_last - timedelta(days=parent_span_days - 1)
    fitted_from = min(value for value in (parent_first, first_day) if value is not None)
    parent_days = parent.support_days if parent else 0
    support_days = parent_days + len(candidates)
    span_days = (last_day - fitted_from).days + 1

    if accumulator.branch is Branch.FEATURE:
        _, new_counts = _feature_candidate_support(candidates)
        if not new_counts:
            raise RuntimeError("accepted Feature candidates have invalid support evidence")
        if parent is not None:
            stored_counts = parent.support_json.get("feature_observation_counts")
            if not isinstance(stored_counts, list) or len(stored_counts) != len(new_counts):
                raise RuntimeError(
                    "incremental Feature release requires parent observation counts"
                )
            counts = [
                int(previous) + current
                for previous, current in zip(stored_counts, new_counts, strict=True)
            ]
            parent_observation_days = int(
                parent.support_json.get("feature_observation_days")
                or parent.support_json.get("observations")
                or parent.support_days
            )
        else:
            counts = new_counts
            parent_observation_days = 0
        observation_days = parent_observation_days + len(candidates)
        used_counts = [value for value in counts if value > 0]
        if observation_days < 1 or any(value > observation_days for value in counts):
            raise RuntimeError("Feature observation counts exceed their day denominator")
        coverage = sum(counts) / (observation_days * len(counts))
        support_json = {
            "active_days": support_days,
            "span_days": span_days,
            "active_days_current_role": support_days,
            "coverage": coverage,
            "min_feature_observations": min(used_counts) if used_counts else 0,
            "feature_observation_counts": counts,
            "feature_observation_days": observation_days,
            "feature_dimension": len(counts),
            "first_active_day": fitted_from.isoformat(),
            "last_active_day": last_day.isoformat(),
            "last_active_gap_days": max((release_day - last_day).days, 0),
            "support_source": "accepted_safe_update_candidates",
        }
        return support_json, support_days, 0, coverage, fitted_from

    new_transitions = sum(
        int(candidate.support_contribution_json.get("transitions", 0))
        for candidate in candidates
    )
    support_transitions = (parent.support_transitions if parent else 0) + new_transitions
    support_json = {
        "sequence_days": support_days,
        "transitions": support_transitions,
        "span_days": span_days,
        "sequence_days_current_role": support_days,
        "first_active_day": fitted_from.isoformat(),
        "first_sequence_day": fitted_from.isoformat(),
        "last_active_day": last_day.isoformat(),
        "last_sequence_day": last_day.isoformat(),
        "last_active_gap_days": max((release_day - last_day).days, 0),
        "support_source": "accepted_safe_update_candidates",
    }
    return support_json, support_days, support_transitions, None, fitted_from


def _materialize_accumulator(
    session: Session,
    organization: Organization,
    accumulator_id: uuid.UUID,
    *,
    logical_day: date,
    framework_config: Mapping[str, Any],
    policy: SafeUpdatePolicy,
    actor: str,
    request_id: str,
) -> int:
    accumulator = session.scalar(
        select(PersonalReferenceAccumulator)
        .where(
            PersonalReferenceAccumulator.id == accumulator_id,
            PersonalReferenceAccumulator.organization_id == organization.id,
        )
        .with_for_update()
    )
    if accumulator is None or accumulator.status in {
        PersonalAccumulatorStatus.CLOSED,
        PersonalAccumulatorStatus.COMPROMISED,
    }:
        return 0
    current_assignment = role_assignment_at(
        session,
        accumulator.user_id,
        logical_day - timedelta(days=1),
    )
    current_epoch = (
        role_epoch_anchor(session, current_assignment)
        if current_assignment is not None
        else None
    )
    if current_epoch is None or current_epoch.id != accumulator.role_assignment_id:
        accumulator.status = PersonalAccumulatorStatus.CLOSED
        stranded = list(
            session.scalars(
                select(SafeUpdateCandidate)
                .where(
                    SafeUpdateCandidate.accumulator_id == accumulator.id,
                    SafeUpdateCandidate.status == UpdateStatus.ACCEPTED,
                    SafeUpdateCandidate.policy_version == policy.policy_version,
                    SafeUpdateCandidate.model_version == accumulator.model_version,
                    SafeUpdateCandidate.config_version == accumulator.config_version,
                )
                .with_for_update()
            )
        )
        for candidate in stranded:
            candidate.status = UpdateStatus.REJECTED
            candidate.reason_codes = ["ROLE_EPOCH_CHANGED"]
            candidate.decision_at = utc_now()
            append_audit(
                session,
                organization,
                actor=actor,
                action="safe_update.rejected",
                entity_type="safe_update_candidate",
                entity_id=str(candidate.id),
                request_id=request_id,
                after={
                    "status": candidate.status.value,
                    "reason_codes": candidate.reason_codes,
                    "decision_at": candidate.decision_at,
                    "policy_version": candidate.policy_version,
                },
            )
            session.flush()
        append_audit(
            session,
            organization,
            actor=actor,
            action="safe_update.accumulator_closed",
            entity_type="personal_reference_accumulator",
            entity_id=str(accumulator.id),
            request_id=request_id,
            after={"reason_code": "ROLE_EPOCH_CHANGED", "logical_day": logical_day},
        )
        session.flush()
        return 0
    if (
        accumulator.last_release_day is not None
        and (logical_day - accumulator.last_release_day).days < policy.release_interval_days
    ):
        return 0

    candidates = list(
        session.scalars(
            select(SafeUpdateCandidate)
            .where(
                SafeUpdateCandidate.accumulator_id == accumulator.id,
                SafeUpdateCandidate.status == UpdateStatus.ACCEPTED,
                SafeUpdateCandidate.policy_version == policy.policy_version,
                SafeUpdateCandidate.model_version == accumulator.model_version,
                SafeUpdateCandidate.config_version == accumulator.config_version,
                SafeUpdateCandidate.eligible_on.is_not(None),
                SafeUpdateCandidate.eligible_on <= logical_day,
            )
            .order_by(SafeUpdateCandidate.candidate_day, SafeUpdateCandidate.id)
            .with_for_update()
        )
    )
    revalidated: list[SafeUpdateCandidate] = []
    for candidate in candidates:
        if not _quarantine_watermarks_complete(
            session,
            organization,
            candidate,
            policy,
        ):
            continue
        reasons = _candidate_integrity_reasons(
            session,
            organization,
            candidate,
            latest_watermark_day=logical_day - timedelta(days=1),
            policy=policy,
        )
        if not reasons:
            revalidated.append(candidate)
            continue
        candidate.status = UpdateStatus.REJECTED
        candidate.reason_codes = ["PRE_MATERIALIZATION_REVALIDATION_FAILED", *reasons]
        candidate.decision_at = utc_now()
        append_audit(
            session,
            organization,
            actor=actor,
            action="safe_update.rejected",
            entity_type="safe_update_candidate",
            entity_id=str(candidate.id),
            request_id=request_id,
            after={
                "status": candidate.status.value,
                "reason_codes": candidate.reason_codes,
                "decision_at": candidate.decision_at,
                "policy_version": policy.policy_version,
            },
        )
        session.flush()
    candidates = revalidated
    session.flush()
    if not candidates:
        return 0

    parent = (
        session.get(ReferenceProfile, accumulator.active_reference_profile_id)
        if accumulator.active_reference_profile_id
        else None
    )
    parent_personal_scores = _profile_personal_scores(parent) if parent else []
    effective_parent_support = len(parent_personal_scores)
    recent_release_rows: list[ReferenceRelease] = []
    rolling_anchor_support = effective_parent_support
    rolling_applied_before = 0
    if parent is None:
        chosen = _bootstrap_candidate_prefix(
            candidates,
            branch=accumulator.branch,
            framework_config=framework_config,
        )
        if not chosen:
            return 0
        release_kind = ReferenceReleaseKind.BOOTSTRAP
        parent_support = 0
        influence_ratio = 1.0
    else:
        recent_release_rows = list(
            session.scalars(
                select(ReferenceRelease).where(
                    ReferenceRelease.accumulator_id == accumulator.id,
                    ReferenceRelease.kind == ReferenceReleaseKind.INCREMENTAL,
                    ReferenceRelease.release_day
                    >= logical_day - timedelta(days=policy.rolling_window_days - 1),
                    ReferenceRelease.release_day <= logical_day,
                )
            )
        )
        capacity = bounded_release_capacity(
            logical_day=logical_day,
            parent_support=effective_parent_support,
            recent_releases=[
                ReleaseWindowEntry(
                    release_day=item.release_day,
                    parent_support=item.parent_support,
                    applied_candidate_count=item.applied_candidate_count,
                )
                for item in recent_release_rows
            ],
            policy=policy,
        )
        if capacity < 1:
            return 0
        chosen = candidates[:capacity]
        release_kind = ReferenceReleaseKind.INCREMENTAL
        parent_support = effective_parent_support
        influence_ratio = len(chosen) / effective_parent_support
        rolling_anchor_support = min(
            [parent_support, *(item.parent_support for item in recent_release_rows)]
        )
        rolling_applied_before = sum(
            item.applied_candidate_count for item in recent_release_rows
        )

    calibration_parent = session.get(
        ReferenceProfile,
        accumulator.admission_reference_profile_id,
    )
    if (
        calibration_parent is None
        or calibration_parent.level not in {ReferenceLevel.ROLE, ReferenceLevel.GLOBAL}
        or not calibration_parent.is_frozen
    ):
        raise RuntimeError("accumulator calibration parent is missing or invalid")
    new_scores = _candidate_raw_scores(session, chosen)
    personal_scores = [*parent_personal_scores, *new_scores]
    personal_scores.sort()
    support_json, support_days, support_transitions, coverage, fitted_from = (
        _child_support_snapshot(
            accumulator=accumulator,
            parent=parent,
            candidates=chosen,
            release_day=logical_day,
        )
    )
    release_sequence = accumulator.release_sequence + 1
    calibrator = build_personal_calibrator(
        personal_scores,
        parent_reference_profile_id=str(calibration_parent.id),
        parent_reference_checksum=calibration_parent.checksum,
        standalone_min_safe_scores=policy.standalone_min_safe_scores,
    )
    statistics = {
        **robust_score_statistics(personal_scores),
        "release_sequence": release_sequence,
        "release_kind": release_kind.value,
        "policy_version": policy.policy_version,
    }
    manifest = {
        "accumulator_id": str(accumulator.id),
        "release_sequence": release_sequence,
        "candidate_ids": [str(item.id) for item in chosen],
        "candidate_days": [item.candidate_day.isoformat() for item in chosen],
        "source_branch_score_ids": [
            str(item.source_branch_score_id) for item in chosen
        ],
        "source_input_checksums": [item.source_input_checksum for item in chosen],
        "admission_reference_profile_ids": [
            str(item.admission_reference_profile_id) for item in chosen
        ],
        "admission_percentiles": [item.admission_percentile for item in chosen],
        "policy_version": policy.policy_version,
    }
    checksum = sha256_json(
        {
            "organization_id": str(organization.id),
            "branch": accumulator.branch.value,
            "scope_key": f"person:{accumulator.role_assignment_id}",
            "model_version": accumulator.model_version,
            "config_version": accumulator.config_version,
            "catalog_version": accumulator.catalog_version,
            "fitted_from": fitted_from,
            "fitted_through": chosen[-1].candidate_day,
            "support": support_json,
            "statistics": statistics,
            "calibrator": calibrator,
            "parent_reference_profile_id": str(parent.id) if parent else None,
            "calibration_parent_profile_id": str(calibration_parent.id),
            "release_day": logical_day,
            "manifest": manifest,
        }
    )
    child = ReferenceProfile(
        organization_id=organization.id,
        branch=accumulator.branch,
        level=ReferenceLevel.PERSON,
        scope_key=f"person:{accumulator.role_assignment_id}",
        user_id=accumulator.user_id,
        role_assignment_id=accumulator.role_assignment_id,
        model_version=accumulator.model_version,
        config_version=accumulator.config_version,
        catalog_version=accumulator.catalog_version,
        fitted_from=fitted_from,
        fitted_through=chosen[-1].candidate_day,
        support_days=support_days,
        support_users=1,
        support_transitions=support_transitions,
        coverage=coverage,
        support_json=support_json,
        statistics_json=statistics,
        calibrator_json=calibrator,
        parent_reference_profile_id=parent.id if parent else None,
        calibration_parent_profile_id=calibration_parent.id,
        reference_version=release_sequence,
        release_kind=release_kind,
        release_day=logical_day,
        release_influence_ratio=influence_ratio,
        policy_version=policy.policy_version,
        is_frozen=False,
        checksum=checksum,
    )
    session.add(child)
    session.flush()

    release = ReferenceRelease(
        organization_id=organization.id,
        accumulator_id=accumulator.id,
        release_sequence=release_sequence,
        kind=release_kind,
        parent_reference_profile_id=parent.id if parent else None,
        calibration_parent_profile_id=calibration_parent.id,
        child_reference_profile_id=child.id,
        release_day=logical_day,
        parent_support=parent_support,
        applied_candidate_count=len(chosen),
        influence_ratio=influence_ratio,
        rolling_anchor_support=rolling_anchor_support,
        rolling_applied_count_before=rolling_applied_before,
        policy_version=policy.policy_version,
        candidate_manifest_json=manifest,
        manifest_checksum=sha256_json(manifest),
        released_at=utc_now(),
    )
    session.add(release)
    session.flush()

    now = utc_now()
    for candidate in chosen:
        candidate.status = UpdateStatus.APPLIED
        candidate.reason_codes = [*candidate.reason_codes, "IMMUTABLE_REFERENCE_RELEASED"]
        candidate.applied_at = now
        candidate.before_checksum = (
            parent.checksum if parent else calibration_parent.checksum
        )
        candidate.after_checksum = child.checksum
        candidate.materialized_reference_profile_id = child.id
        candidate.reference_release_id = release.id
    accumulator.active_reference_profile_id = child.id
    accumulator.status = PersonalAccumulatorStatus.ACTIVE
    accumulator.release_sequence = release_sequence
    accumulator.last_release_day = logical_day
    session.flush()
    append_audit(
        session,
        organization,
        actor=actor,
        action="safe_update.reference_released",
        entity_type="reference_release",
        entity_id=str(release.id),
        request_id=request_id,
        after={
            "accumulator_id": str(accumulator.id),
            "child_reference_profile_id": str(child.id),
            "release_sequence": release_sequence,
            "release_kind": release_kind.value,
            "applied_candidate_count": len(chosen),
            "influence_ratio": influence_ratio,
            "manifest_checksum": release.manifest_checksum,
        },
    )
    return len(chosen)


def process_safe_updates(
    session: Session,
    organization: Organization,
    *,
    model_version: str,
    config_version: str,
    limit: int,
    actor: str,
    request_id: str,
) -> dict[str, int]:
    framework_config = load_framework_config()
    if config_version != str(framework_config["schema_version"]):
        raise DomainValidationError(
            "SAFE_UPDATE_CONFIG_NOT_ACTIVE",
            "safe-update processing can only target the active v5 framework release",
            {
                "active": str(framework_config["schema_version"]),
                "provided": config_version,
            },
        )
    policy = SafeUpdatePolicy.from_framework(framework_config)
    rejected_before = int(
        session.scalar(
            select(func.count(SafeUpdateCandidate.id)).where(
                SafeUpdateCandidate.organization_id == organization.id,
                SafeUpdateCandidate.status == UpdateStatus.REJECTED,
                SafeUpdateCandidate.policy_version == policy.policy_version,
                SafeUpdateCandidate.model_version == model_version,
                SafeUpdateCandidate.config_version == config_version,
            )
        )
        or 0
    )
    latest_watermark_day = session.scalar(
        select(func.max(ScoringWatermark.day)).where(
            ScoringWatermark.organization_id == organization.id,
            ScoringWatermark.model_version == model_version,
            ScoringWatermark.config_version == config_version,
        )
    )
    accepted = 0
    rejected = 0
    applied = 0
    if latest_watermark_day is not None:
        logical_day = latest_watermark_day + timedelta(days=1)
        candidates = list(
            session.scalars(
                select(SafeUpdateCandidate)
                .where(
                    SafeUpdateCandidate.organization_id == organization.id,
                    SafeUpdateCandidate.status == UpdateStatus.CANDIDATE,
                    SafeUpdateCandidate.policy_version == policy.policy_version,
                    SafeUpdateCandidate.model_version == model_version,
                    SafeUpdateCandidate.config_version == config_version,
                    SafeUpdateCandidate.eligible_on.is_not(None),
                    SafeUpdateCandidate.eligible_on <= logical_day,
                )
                .order_by(
                    SafeUpdateCandidate.eligible_on,
                    SafeUpdateCandidate.candidate_day,
                    SafeUpdateCandidate.id,
                )
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
        )
        for candidate in candidates:
            if not _quarantine_watermarks_complete(
                session,
                organization,
                candidate,
                policy,
            ):
                continue
            reasons = _candidate_integrity_reasons(
                session,
                organization,
                candidate,
                latest_watermark_day=latest_watermark_day,
                policy=policy,
            )
            candidate.decision_at = utc_now()
            if reasons:
                candidate.status = UpdateStatus.REJECTED
                candidate.reason_codes = reasons
                rejected += 1
            else:
                candidate.status = UpdateStatus.ACCEPTED
                candidate.reason_codes = ["QUARANTINE_AND_WATERMARKS_PASSED"]
                accepted += 1
            append_audit(
                session,
                organization,
                actor=actor,
                action=f"safe_update.{candidate.status.value}",
                entity_type="safe_update_candidate",
                entity_id=str(candidate.id),
                request_id=request_id,
                after={
                    "status": candidate.status.value,
                    "reason_codes": candidate.reason_codes,
                    "decision_at": candidate.decision_at,
                    "policy_version": policy.policy_version,
                },
            )
            session.flush()
        session.flush()

        accumulator_ids = list(
            session.scalars(
                select(SafeUpdateCandidate.accumulator_id)
                .where(
                    SafeUpdateCandidate.organization_id == organization.id,
                    SafeUpdateCandidate.status == UpdateStatus.ACCEPTED,
                    SafeUpdateCandidate.policy_version == policy.policy_version,
                    SafeUpdateCandidate.model_version == model_version,
                    SafeUpdateCandidate.config_version == config_version,
                    SafeUpdateCandidate.accumulator_id.is_not(None),
                )
                .distinct()
            )
        )
        for accumulator_id in sorted(accumulator_ids, key=str):
            if accumulator_id is None:
                continue
            applied += _materialize_accumulator(
                session,
                organization,
                accumulator_id,
                logical_day=logical_day,
                framework_config=framework_config,
                policy=policy,
                actor=actor,
                request_id=request_id,
            )

    rejected_after = int(
        session.scalar(
            select(func.count(SafeUpdateCandidate.id)).where(
                SafeUpdateCandidate.organization_id == organization.id,
                SafeUpdateCandidate.status == UpdateStatus.REJECTED,
                SafeUpdateCandidate.policy_version == policy.policy_version,
                SafeUpdateCandidate.model_version == model_version,
                SafeUpdateCandidate.config_version == config_version,
            )
        )
        or 0
    )
    rejected = max(rejected_after - rejected_before, 0)

    pending = int(
        session.scalar(
            select(func.count(SafeUpdateCandidate.id)).where(
                SafeUpdateCandidate.organization_id == organization.id,
                SafeUpdateCandidate.status == UpdateStatus.CANDIDATE,
                SafeUpdateCandidate.policy_version == policy.policy_version,
                SafeUpdateCandidate.model_version == model_version,
                SafeUpdateCandidate.config_version == config_version,
            )
        )
        or 0
    )
    deferred = int(
        session.scalar(
            select(func.count(SafeUpdateCandidate.id)).where(
                SafeUpdateCandidate.organization_id == organization.id,
                SafeUpdateCandidate.status == UpdateStatus.ACCEPTED,
                SafeUpdateCandidate.policy_version == policy.policy_version,
                SafeUpdateCandidate.model_version == model_version,
                SafeUpdateCandidate.config_version == config_version,
            )
        )
        or 0
    )
    legacy_pending = int(
        session.scalar(
            select(func.count(SafeUpdateCandidate.id)).where(
                SafeUpdateCandidate.organization_id == organization.id,
                SafeUpdateCandidate.status.in_(
                    [UpdateStatus.CANDIDATE, UpdateStatus.ACCEPTED]
                ),
                SafeUpdateCandidate.policy_version != policy.policy_version,
            )
        )
        or 0
    )
    return {
        "accepted": accepted,
        "rejected": rejected,
        "applied": applied,
        "deferred": deferred,
        "pending": pending,
        "legacy_pending": legacy_pending,
    }


def framework_info() -> dict[str, Any]:
    catalog = load_feature_catalog()
    config = load_framework_config()
    sequence = load_sequence_config()
    return {
        "feature_schema_version": catalog["schema_version"],
        "feature_dimension": len(catalog["features"]),
        "sequence_schema_version": sequence["schema_version"],
        "sequence_tokens": sequence["tokens"],
        "framework_config": config,
    }
