from __future__ import annotations

import uuid
from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.catalog import (
    FEATURE_SCHEMA_VERSION,
    FRAMEWORK_SCHEMA_VERSION,
    SEQUENCE_SCHEMA_VERSION,
    load_feature_catalog,
)
from app.database import Base, get_db
from app.errors import ApiError, not_found
from app.models import (
    Artifact,
    FeatureCatalog,
    FeatureDefinition,
    Organization,
    ReferenceProfile,
    Role,
    RoleAssignment,
)
from app.schemas import (
    AlertPatch,
    AlertRead,
    AlertStatusFilter,
    ArtifactCreate,
    ArtifactRead,
    AssessmentCreate,
    AssessmentDetail,
    AssessmentRead,
    CheckpointRead,
    CheckpointUpsert,
    EventBatchCreate,
    EventBatchResult,
    EventRead,
    FeatureVectorRead,
    FeatureVectorUpsert,
    FrameworkInfo,
    IngestionJobCreate,
    IngestionJobRead,
    ReferenceProfileCreate,
    ReferenceProfileRead,
    RoleAssignmentCreate,
    RoleAssignmentRead,
    RoleCreate,
    RoleRead,
    SafeUpdateProcessRequest,
    SafeUpdateProcessResult,
    ScoringWatermarkCloseRequest,
    ScoringWatermarkRead,
    SequenceRead,
    SequenceUpsert,
    SourceType,
    UserCreate,
    UserRead,
)
from app.services import (
    assign_role,
    close_scoring_day,
    create_artifact,
    create_assessment,
    create_ingestion_job,
    create_reference_profile,
    create_role,
    create_user,
    ensure_default_organization,
    framework_info,
    get_assessment,
    get_feature_vector,
    get_ingestion_job,
    get_sequence,
    get_user,
    ingest_events,
    list_alerts,
    list_events,
    patch_alert,
    process_safe_updates,
    role_assignment_at,
    upsert_checkpoint,
    upsert_feature_vector,
    upsert_sequence,
)
from app.web import require_scorer_api_key

DbSession = Annotated[Session, Depends(get_db)]
router = APIRouter()
api = APIRouter(prefix="/api/v1")


def _commit(session: Session) -> None:
    try:
        session.commit()
    except Exception:
        session.rollback()
        raise


def _request_context(request: Request) -> tuple[str, str]:
    return request.state.actor, request.state.request_id


def _organization(session: Session) -> Organization:
    organization = ensure_default_organization(session)
    if organization in session.new:
        _commit(session)
    return organization


def _user_response(user: Any) -> dict[str, Any]:
    return {
        "user_id": user.external_user_id,
        "display_name": user.display_name,
        "status": user.status.value.upper(),
        "metadata_json": user.attributes_json,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
    }


def _role_response(role: Role) -> dict[str, Any]:
    return {
        "role_code": role.code,
        "display_name": role.name,
        "role_family": role.family,
        "metadata_json": role.attributes_json,
        "created_at": role.created_at,
    }


def _assignment_response(
    assignment: RoleAssignment, external_user_id: str, role_code: str
) -> dict[str, Any]:
    return {
        "id": str(assignment.id),
        "user_id": external_user_id,
        "role_code": role_code,
        "valid_from": assignment.valid_from,
        "valid_to": assignment.valid_to,
        "source_snapshot_date": assignment.source_snapshot_date,
        "created_at": assignment.created_at,
    }


def _job_response(job: Any) -> dict[str, Any]:
    return {
        "id": str(job.id),
        "source": job.source.value.upper(),
        "status": job.status.value.upper(),
        "input_uri": job.input_uri,
        "original_filename": job.original_filename,
        "sha256": job.sha256,
        "idempotency_key": job.idempotency_key,
        "schema_version": job.schema_version,
        "total_rows": job.total_rows,
        "processed_rows": job.processed_rows,
        "rejected_rows": job.rejected_rows,
        "progress_json": job.progress_json,
        "error_message": job.error_message,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def _checkpoint_response(checkpoint: Any) -> dict[str, Any]:
    return {
        "id": str(checkpoint.id),
        "job_id": str(checkpoint.job_id),
        "partition_key": checkpoint.partition_key,
        "row_offset": checkpoint.row_offset,
        "cursor_json": checkpoint.cursor_json,
        "input_checksum": checkpoint.input_checksum,
        "output_checksum": checkpoint.output_checksum,
        "created_at": checkpoint.created_at,
        "updated_at": checkpoint.updated_at,
    }


def _event_response(event: Any, external_user_id: str) -> dict[str, Any]:
    return {
        "event_uid": event.event_uid,
        "source": event.source.value.upper(),
        "original_id": event.original_id,
        "timestamp": event.event_timestamp,
        "event_date": event.event_date,
        "user_id": external_user_id,
        "pc": event.pc,
        "action": event.action,
        "object_ref": event.object_ref,
        "pc_context": event.pc_context.value.upper(),
        "source_payload_json": event.source_payload,
        "ingestion_job_id": str(event.ingestion_job_id),
        "created_at": event.created_at,
    }


def _feature_response(
    vector: Any,
    external_user_id: str,
    schema_version: str,
    values: dict[str, float | None],
    masks: dict[str, bool],
) -> dict[str, Any]:
    return {
        "id": str(vector.id),
        "user_id": external_user_id,
        "day": vector.day,
        "schema_version": schema_version,
        "values_json": values,
        "masks_json": masks,
        "context_json": vector.context_json,
        "input_checksum": vector.input_checksum,
        "created_at": vector.created_at,
        "updated_at": vector.updated_at,
    }


def _sequence_response(sequence: Any, external_user_id: str) -> dict[str, Any]:
    return {
        "id": str(sequence.id),
        "user_id": external_user_id,
        "day": sequence.day,
        "schema_version": sequence.vocabulary_version,
        "tokens_json": sequence.tokens,
        "pc_contexts_json": sequence.pc_contexts,
        "calendar_contexts_json": sequence.calendar_contexts,
        "gap_buckets_json": sequence.gap_buckets,
        "event_uids_json": sequence.event_uids,
        "side_fields_json": sequence.side_fields,
        "time_sin_json": [
            float(values.get("time_sin", 0.0)) for values in sequence.side_fields
        ],
        "time_cos_json": [
            float(values.get("time_cos", 0.0)) for values in sequence.side_fields
        ],
        "seq_len": sequence.seq_len,
        "truncated": sequence.truncated,
        "max_len": 256,
        "input_checksum": sequence.input_checksum,
        "created_at": sequence.created_at,
        "updated_at": sequence.updated_at,
    }


_ARTIFACT_API_NAMES = {
    "canonical_events": "00_events",
    "features": "01_feature128",
    "sequences": "02_sequences",
    "role_context": "03_role_context",
    "branch_scores": "04_branch_scores",
    "reference_profile": "05_reference_tables",
    "risk_assessments": "06_risk_alerts",
}


def _artifact_response(artifact: Artifact) -> dict[str, Any]:
    return {
        "id": str(artifact.id),
        "artifact_type": _ARTIFACT_API_NAMES.get(artifact.kind.value, artifact.kind.value),
        "uri": artifact.uri,
        "schema_version": artifact.schema_version or "unknown",
        "checksum": artifact.checksum,
        "row_count": artifact.row_count,
        "min_date": artifact.min_date,
        "max_date": artifact.max_date,
        "status": artifact.status.value.upper(),
        "metadata_json": artifact.manifest_json,
        "ingestion_job_id": (str(artifact.ingestion_job_id) if artifact.ingestion_job_id else None),
        "created_at": artifact.created_at,
    }


def _reference_response(profile: ReferenceProfile) -> dict[str, Any]:
    support = dict(profile.support_json)
    return {
        "id": str(profile.id),
        "branch": profile.branch.value.upper(),
        "level": profile.level.value.upper(),
        "scope_key": profile.scope_key,
        "as_of_date": date.fromisoformat(
            support.get("_as_of_date", (profile.fitted_through + date.resolution).isoformat())
        ),
        "model_version": profile.model_version,
        "config_version": profile.config_version,
        "version": support.get("_reference_version", profile.checksum[:12]),
        "fitted_through": profile.fitted_through,
        "frozen": profile.is_frozen,
        "support_json": support,
        "statistics_json": profile.statistics_json,
        "parent_reference_profile_id": (
            str(profile.parent_reference_profile_id)
            if profile.parent_reference_profile_id
            else None
        ),
        "calibration_parent_profile_id": (
            str(profile.calibration_parent_profile_id)
            if profile.calibration_parent_profile_id
            else None
        ),
        "reference_version": profile.reference_version,
        "release_kind": profile.release_kind.value.upper() if profile.release_kind else None,
        "release_day": profile.release_day,
        "release_influence_ratio": profile.release_influence_ratio,
        "policy_version": profile.policy_version,
        "checksum": profile.checksum,
        "created_at": profile.created_at,
    }


def _branch_response(score: Any | None) -> dict[str, Any] | None:
    if score is None:
        return None
    return {
        "id": str(score.id),
        "branch": score.branch.value.upper(),
        "raw_score": score.raw_score,
        "calibrated_score": score.calibrated_score,
        "selected_level": (
            score.selected_level.value.upper() if score.selected_level is not None else "NO_SCORE"
        ),
        "reference_profile_id": (
            str(score.reference_profile_id) if score.reference_profile_id else None
        ),
        "fallback_reasons_json": score.fallback_reasons,
        "support_json": score.support_snapshot,
        "evidence_json": score.evidence,
        "model_version": score.model_version,
        "config_version": score.config_version,
        "created_at": score.created_at,
    }


def _assessment_response(
    assessment: Any, external_user_id: str, role_code: str | None
) -> dict[str, Any]:
    return {
        "id": str(assessment.id),
        "user_id": external_user_id,
        "day": assessment.day,
        "role_code": role_code,
        "split": assessment.split.value.upper(),
        "model_version": assessment.model_version,
        "config_version": assessment.config_version,
        "feature_score_id": (
            str(assessment.feature_score_id) if assessment.feature_score_id else None
        ),
        "sequence_score_id": (
            str(assessment.sequence_score_id) if assessment.sequence_score_id else None
        ),
        "feature_weight": assessment.feature_weight,
        "sequence_weight": assessment.sequence_weight,
        "risk": assessment.risk,
        "threshold": assessment.threshold,
        "state": assessment.status.value.upper(),
        "is_alert": assessment.is_alert,
        "no_score_reason": ("BOTH_BRANCHES_NO_SCORE" if assessment.risk is None else None),
        "created_at": assessment.created_at,
    }


def _alert_response(alert: Any, assessment: Any, external_user_id: str) -> dict[str, Any]:
    return {
        "id": str(alert.id),
        "assessment_id": str(assessment.id),
        "user_id": external_user_id,
        "day": assessment.day,
        "risk": assessment.risk,
        "threshold": assessment.threshold,
        "severity": alert.severity.value.upper(),
        "status": alert.status.value.upper(),
        "assignee": alert.assignee,
        "resolution": alert.resolution,
        "opened_at": alert.opened_at,
        "updated_at": alert.updated_at,
        "closed_at": alert.closed_at,
    }


def _safe_update_response(
    candidate: Any, external_user_id: str, role_code: str | None
) -> dict[str, Any]:
    return {
        "id": str(candidate.id),
        "assessment_id": str(candidate.source_assessment_id),
        "user_id": external_user_id,
        "day": candidate.candidate_day,
        "role_code": role_code,
        "branch": candidate.branch.value.upper(),
        "status": candidate.status.value.upper(),
        "quarantine_until": candidate.quarantine_until,
        "eligible_on": candidate.eligible_on,
        "admission_percentile": candidate.admission_percentile,
        "admission_threshold": candidate.admission_threshold,
        "policy_version": candidate.policy_version,
        "materialized_reference_profile_id": (
            str(candidate.materialized_reference_profile_id)
            if candidate.materialized_reference_profile_id
            else None
        ),
        "reference_release_id": (
            str(candidate.reference_release_id)
            if candidate.reference_release_id
            else None
        ),
        "decision_reasons_json": candidate.reason_codes,
        "decided_at": (
            candidate.decision_at or candidate.updated_at
            if candidate.status.value in {"accepted", "rejected", "applied"}
            else None
        ),
        "created_at": candidate.created_at,
    }


def _scoring_watermark_response(watermark: Any) -> dict[str, Any]:
    return {
        "id": str(watermark.id),
        "day": watermark.day,
        "model_version": watermark.model_version,
        "config_version": watermark.config_version,
        "expected_assessments": watermark.expected_assessments,
        "persisted_assessments": watermark.persisted_assessments,
        "universe_checksum": watermark.universe_checksum,
        "assessment_set_checksum": watermark.assessment_set_checksum,
        "completed_at": watermark.completed_at,
    }


@router.get("/health/live", tags=["health"])
def live() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready", tags=["health"])
def ready(request: Request, db: DbSession) -> dict[str, str]:
    try:
        available_tables = set(inspect(db.get_bind()).get_table_names())
    except SQLAlchemyError as exc:
        raise ApiError(
            503,
            "DATABASE_NOT_READY",
            "database connection is not ready",
        ) from exc

    missing_tables = sorted(set(Base.metadata.tables) - available_tables)
    if missing_tables:
        raise ApiError(
            503,
            "DATABASE_SCHEMA_NOT_READY",
            "required database tables are missing",
            {"missing_tables": missing_tables},
        )

    runtime_settings = request.app.state.settings
    try:
        feature_config = runtime_settings.load_feature_catalog(required=True)
        framework_config = runtime_settings.load_framework_config(required=True)
        sequence_config = runtime_settings.load_sequence_config(required=True)
    except (OSError, ValueError) as exc:
        raise ApiError(
            503,
            "FRAMEWORK_CONFIG_NOT_READY",
            "locked framework configuration is missing or invalid",
        ) from exc

    features = feature_config.get("features")
    tokens = sequence_config.get("tokens")
    if (
        feature_config.get("schema_version") != FEATURE_SCHEMA_VERSION
        or not isinstance(features, list)
        or len(features) != 128
        or framework_config.get("schema_version") != FRAMEWORK_SCHEMA_VERSION
        or sequence_config.get("schema_version") != SEQUENCE_SCHEMA_VERSION
        or not isinstance(tokens, list)
        or len(tokens) != 7
    ):
        raise ApiError(
            503,
            "FRAMEWORK_CONFIG_NOT_READY",
            "locked framework configuration is missing or invalid",
        )

    schema_version = str(feature_config["schema_version"])
    catalog_name = str(feature_config.get("feature_set") or schema_version)
    catalog_version = str(feature_config.get("feature_version") or schema_version)
    catalog = db.scalar(
        select(FeatureCatalog).where(
            FeatureCatalog.name == catalog_name,
            FeatureCatalog.version == catalog_version,
            FeatureCatalog.dimension == 128,
            FeatureCatalog.is_active.is_(True),
        )
    )
    definition_count = (
        db.scalar(
            select(func.count(FeatureDefinition.id)).where(
                FeatureDefinition.catalog_id == catalog.id
            )
        )
        if catalog is not None
        else 0
    )
    if catalog is None or definition_count != 128:
        raise ApiError(
            503,
            "FEATURE_CATALOG_NOT_READY",
            "an active 128-feature catalog is not initialized",
        )
    return {"status": "ready"}


@api.get("/framework", response_model=FrameworkInfo, tags=["framework"])
def get_framework() -> dict[str, Any]:
    return framework_info()


@api.get("/features/catalog", tags=["framework"])
def get_catalog() -> dict[str, Any]:
    return load_feature_catalog()


@api.post(
    "/users",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    tags=["identity"],
)
def post_user(payload: UserCreate, request: Request, db: DbSession) -> dict[str, Any]:
    organization = _organization(db)
    actor, request_id = _request_context(request)
    user = create_user(db, organization, payload, actor=actor, request_id=request_id)
    _commit(db)
    return _user_response(user)


@api.get("/users/{user_id}", response_model=UserRead, tags=["identity"])
def read_user(user_id: str, db: DbSession) -> dict[str, Any]:
    organization = _organization(db)
    return _user_response(get_user(db, organization, user_id))


@api.post(
    "/roles",
    response_model=RoleRead,
    status_code=status.HTTP_201_CREATED,
    tags=["identity"],
)
def post_role(payload: RoleCreate, request: Request, db: DbSession) -> dict[str, Any]:
    organization = _organization(db)
    actor, request_id = _request_context(request)
    role = create_role(db, organization, payload, actor=actor, request_id=request_id)
    _commit(db)
    return _role_response(role)


@api.post(
    "/users/{user_id}/role-assignments",
    response_model=RoleAssignmentRead,
    status_code=status.HTTP_201_CREATED,
    tags=["identity"],
)
def post_role_assignment(
    user_id: str,
    payload: RoleAssignmentCreate,
    request: Request,
    db: DbSession,
) -> dict[str, Any]:
    organization = _organization(db)
    actor, request_id = _request_context(request)
    user = get_user(db, organization, user_id)
    assignment = assign_role(
        db,
        organization,
        user,
        payload,
        actor=actor,
        request_id=request_id,
    )
    _commit(db)
    return _assignment_response(assignment, user_id, payload.role_code)


@api.get(
    "/users/{user_id}/role",
    response_model=RoleAssignmentRead,
    tags=["identity"],
)
def read_current_role(user_id: str, db: DbSession, as_of: date = Query(...)) -> dict[str, Any]:
    organization = _organization(db)
    user = get_user(db, organization, user_id)
    assignment = role_assignment_at(db, user.id, as_of)
    if assignment is None:
        raise not_found("role assignment", f"{user_id}/{as_of}")
    role = db.get(Role, assignment.role_id)
    return _assignment_response(assignment, user_id, role.code)


@api.post(
    "/ingestions",
    response_model=IngestionJobRead,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["ingestion"],
)
def post_ingestion(payload: IngestionJobCreate, request: Request, db: DbSession) -> dict[str, Any]:
    organization = _organization(db)
    actor, request_id = _request_context(request)
    job = create_ingestion_job(db, organization, payload, actor=actor, request_id=request_id)
    _commit(db)
    return _job_response(job)


@api.get("/ingestions/{job_id}", response_model=IngestionJobRead, tags=["ingestion"])
def read_ingestion(job_id: str, db: DbSession) -> dict[str, Any]:
    organization = _organization(db)
    return _job_response(get_ingestion_job(db, organization, job_id))


@api.put(
    "/ingestions/{job_id}/checkpoints",
    response_model=CheckpointRead,
    tags=["ingestion"],
)
def put_checkpoint(
    job_id: str,
    payload: CheckpointUpsert,
    request: Request,
    db: DbSession,
) -> dict[str, Any]:
    organization = _organization(db)
    actor, request_id = _request_context(request)
    job = get_ingestion_job(db, organization, job_id)
    checkpoint = upsert_checkpoint(
        db,
        organization,
        job,
        payload,
        actor=actor,
        request_id=request_id,
    )
    _commit(db)
    return _checkpoint_response(checkpoint)


@api.get(
    "/ingestions/{job_id}/checkpoints",
    response_model=list[CheckpointRead],
    tags=["ingestion"],
)
def read_checkpoints(job_id: str, db: DbSession) -> list[dict[str, Any]]:
    organization = _organization(db)
    job = get_ingestion_job(db, organization, job_id)
    return [
        _checkpoint_response(checkpoint)
        for checkpoint in sorted(job.checkpoints, key=lambda item: item.partition_key)
    ]


@api.post(
    "/events/batch",
    response_model=EventBatchResult,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["events"],
)
def post_events(payload: EventBatchCreate, request: Request, db: DbSession) -> dict[str, Any]:
    configured_limit = min(int(request.app.state.settings.max_batch_events), 1000)
    if len(payload.events) > configured_limit:
        raise ApiError(
            413,
            "EVENT_BATCH_TOO_LARGE",
            f"event batch exceeds configured limit of {configured_limit}",
            {"configured_limit": configured_limit, "received": len(payload.events)},
        )
    organization = _organization(db)
    actor, request_id = _request_context(request)
    inserted, duplicates, event_uids = ingest_events(
        db,
        organization,
        payload,
        actor=actor,
        request_id=request_id,
    )
    _commit(db)
    return {
        "inserted": inserted,
        "duplicates": duplicates,
        "event_uids": event_uids,
    }


@api.get("/events", response_model=list[EventRead], tags=["events"])
def read_events(
    db: DbSession,
    user_id: str | None = None,
    day: date | None = None,
    source: SourceType | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    organization = _organization(db)
    rows = list_events(
        db,
        organization,
        external_user_id=user_id,
        day=day,
        source=source.value if source else None,
        limit=limit,
        offset=offset,
    )
    return [_event_response(event, external_id) for event, external_id in rows]


@api.put(
    "/user-days/{user_id}/{day}/features",
    response_model=FeatureVectorRead,
    tags=["user-days"],
    dependencies=[Depends(require_scorer_api_key)],
)
def put_features(
    user_id: str,
    day: date,
    payload: FeatureVectorUpsert,
    request: Request,
    db: DbSession,
) -> dict[str, Any]:
    organization = _organization(db)
    actor, request_id = _request_context(request)
    user = get_user(db, organization, user_id)
    vector = upsert_feature_vector(
        db,
        organization,
        user,
        day,
        payload,
        actor=actor,
        request_id=request_id,
    )
    _commit(db)
    vector, values, masks = get_feature_vector(db, organization, user, day, payload.schema_version)
    return _feature_response(vector, user_id, payload.schema_version, values, masks)


@api.get(
    "/user-days/{user_id}/{day}/features",
    response_model=FeatureVectorRead,
    tags=["user-days"],
)
def read_features(
    user_id: str,
    day: date,
    db: DbSession,
    schema_version: str = FEATURE_SCHEMA_VERSION,
) -> dict[str, Any]:
    organization = _organization(db)
    user = get_user(db, organization, user_id)
    vector, values, masks = get_feature_vector(db, organization, user, day, schema_version)
    return _feature_response(vector, user_id, schema_version, values, masks)


@api.put(
    "/user-days/{user_id}/{day}/sequence",
    response_model=SequenceRead,
    tags=["user-days"],
    dependencies=[Depends(require_scorer_api_key)],
)
def put_sequence(
    user_id: str,
    day: date,
    payload: SequenceUpsert,
    request: Request,
    db: DbSession,
) -> dict[str, Any]:
    organization = _organization(db)
    actor, request_id = _request_context(request)
    user = get_user(db, organization, user_id)
    sequence = upsert_sequence(
        db,
        organization,
        user,
        day,
        payload,
        actor=actor,
        request_id=request_id,
    )
    _commit(db)
    return _sequence_response(sequence, user_id)


@api.get(
    "/user-days/{user_id}/{day}/sequence",
    response_model=SequenceRead,
    tags=["user-days"],
)
def read_sequence(
    user_id: str,
    day: date,
    db: DbSession,
    schema_version: str = SEQUENCE_SCHEMA_VERSION,
) -> dict[str, Any]:
    organization = _organization(db)
    user = get_user(db, organization, user_id)
    sequence = get_sequence(db, organization, user, day, schema_version)
    return _sequence_response(sequence, user_id)


@api.post(
    "/artifacts",
    response_model=ArtifactRead,
    status_code=status.HTTP_201_CREATED,
    tags=["artifacts"],
)
def post_artifact(payload: ArtifactCreate, request: Request, db: DbSession) -> dict[str, Any]:
    organization = _organization(db)
    actor, request_id = _request_context(request)
    artifact = create_artifact(db, organization, payload, actor=actor, request_id=request_id)
    _commit(db)
    return _artifact_response(artifact)


@api.get("/artifacts", response_model=list[ArtifactRead], tags=["artifacts"])
def read_artifacts(
    db: DbSession,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    organization = _organization(db)
    artifacts = db.scalars(
        select(Artifact)
        .where(Artifact.organization_id == organization.id)
        .order_by(Artifact.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return [_artifact_response(item) for item in artifacts]


@api.post(
    "/references",
    response_model=ReferenceProfileRead,
    status_code=status.HTTP_201_CREATED,
    tags=["scoring"],
    dependencies=[Depends(require_scorer_api_key)],
)
def post_reference(
    payload: ReferenceProfileCreate, request: Request, db: DbSession
) -> dict[str, Any]:
    organization = _organization(db)
    actor, request_id = _request_context(request)
    profile = create_reference_profile(
        db, organization, payload, actor=actor, request_id=request_id
    )
    _commit(db)
    return _reference_response(profile)


@api.get(
    "/references/{profile_id}",
    response_model=ReferenceProfileRead,
    tags=["scoring"],
)
def read_reference(profile_id: str, db: DbSession) -> dict[str, Any]:
    organization = _organization(db)
    try:
        parsed = uuid.UUID(profile_id)
    except ValueError as exc:
        raise not_found("reference profile", profile_id) from exc
    profile = db.scalar(
        select(ReferenceProfile).where(
            ReferenceProfile.id == parsed,
            ReferenceProfile.organization_id == organization.id,
        )
    )
    if profile is None:
        raise not_found("reference profile", profile_id)
    return _reference_response(profile)


@api.post(
    "/assessments/score",
    response_model=AssessmentRead,
    status_code=status.HTTP_201_CREATED,
    tags=["scoring"],
    dependencies=[Depends(require_scorer_api_key)],
)
def score_assessment(payload: AssessmentCreate, request: Request, db: DbSession) -> dict[str, Any]:
    organization = _organization(db)
    actor, request_id = _request_context(request)
    assessment = create_assessment(
        db,
        organization,
        payload,
        actor=actor,
        request_id=request_id,
        locked_threshold=request.app.state.settings.locked_alert_threshold,
    )
    _commit(db)
    role_code = None
    if assessment.role_assignment_id:
        assignment = db.get(RoleAssignment, assessment.role_assignment_id)
        role = db.get(Role, assignment.role_id)
        role_code = role.code
    return _assessment_response(assessment, payload.user_id, role_code)


@api.get(
    "/assessments/{user_id}/{day}",
    response_model=AssessmentDetail,
    tags=["scoring"],
)
def read_assessment(
    user_id: str,
    day: date,
    model_version: str,
    config_version: Annotated[str, Query(min_length=1, max_length=128)],
    db: DbSession,
) -> dict[str, Any]:
    organization = _organization(db)
    (
        assessment,
        user,
        role,
        feature,
        sequence,
        alert,
        candidates,
    ) = get_assessment(
        db,
        organization,
        user_id,
        day,
        model_version,
        config_version,
    )
    return {
        "assessment": _assessment_response(
            assessment, user.external_user_id, role.code if role else None
        ),
        "feature": _branch_response(feature),
        "sequence": _branch_response(sequence),
        "alert": (_alert_response(alert, assessment, user.external_user_id) if alert else None),
        "safe_updates": [
            _safe_update_response(candidate, user.external_user_id, role.code if role else None)
            for candidate in candidates
        ],
    }


@api.get("/alerts", response_model=list[AlertRead], tags=["alerts"])
def read_alerts(
    db: DbSession,
    status_filter: AlertStatusFilter | None = Query(default=None, alias="status"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    organization = _organization(db)
    rows = list_alerts(
        db,
        organization,
        status=status_filter.value if status_filter else None,
        limit=limit,
        offset=offset,
    )
    return [
        _alert_response(alert, assessment, user.external_user_id)
        for alert, assessment, user in rows
    ]


@api.patch("/alerts/{alert_id}", response_model=AlertRead, tags=["alerts"])
def update_alert(
    alert_id: str, payload: AlertPatch, request: Request, db: DbSession
) -> dict[str, Any]:
    organization = _organization(db)
    actor, request_id = _request_context(request)
    alert, assessment, user = patch_alert(
        db,
        organization,
        alert_id,
        payload,
        actor=actor,
        request_id=request_id,
    )
    _commit(db)
    return _alert_response(alert, assessment, user.external_user_id)


@api.post(
    "/safe-updates/process",
    response_model=SafeUpdateProcessResult,
    tags=["safe-update"],
    dependencies=[Depends(require_scorer_api_key)],
)
def post_safe_update_process(
    payload: SafeUpdateProcessRequest, request: Request, db: DbSession
) -> dict[str, int]:
    organization = _organization(db)
    actor, request_id = _request_context(request)
    result = process_safe_updates(
        db,
        organization,
        model_version=payload.model_version,
        config_version=payload.config_version,
        runtime_materialization_enabled=(
            request.app.state.settings.safe_update_materialization_enabled
        ),
        limit=payload.limit,
        actor=actor,
        request_id=request_id,
    )
    _commit(db)
    return result


@api.post(
    "/scoring-watermarks/close-day",
    response_model=ScoringWatermarkRead,
    status_code=status.HTTP_201_CREATED,
    tags=["safe-update"],
    dependencies=[Depends(require_scorer_api_key)],
)
def post_close_scoring_day(
    payload: ScoringWatermarkCloseRequest,
    request: Request,
    db: DbSession,
) -> Any:
    organization = _organization(db)
    actor, request_id = _request_context(request)
    watermark = close_scoring_day(
        db,
        organization,
        day=payload.day,
        model_version=payload.model_version,
        config_version=payload.config_version,
        actor=actor,
        request_id=request_id,
    )
    _commit(db)
    return _scoring_watermark_response(watermark)


router.include_router(api)
