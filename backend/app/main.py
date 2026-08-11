from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Engine

from app.api import router
from app.database import engine, init_db, session_scope
from app.services import ensure_default_organization
from app.settings import Settings, settings
from app.web import RequestContextMiddleware, register_exception_handlers


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def validate_security_settings(app_settings: Settings) -> None:
    environment = app_settings.app_env.strip().lower()
    if environment not in {"prod", "production"}:
        return

    database_scheme = app_settings.database_url.partition(":")[0].strip().lower()
    if (
        app_settings.safe_update_materialization_enabled
        and database_scheme != "postgresql"
        and not database_scheme.startswith("postgresql+")
    ):
        raise RuntimeError(
            "production SAFE_UPDATE_MATERIALIZATION_ENABLED requires a "
            "PostgreSQL DATABASE_URL"
        )

    missing: list[str] = []
    if not app_settings.api_key or not app_settings.api_key.strip():
        missing.append("API_KEY")
    if not app_settings.scorer_api_key or not app_settings.scorer_api_key.strip():
        missing.append("SCORER_API_KEY")
    if app_settings.locked_alert_threshold is None:
        missing.append("LOCKED_ALERT_THRESHOLD")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"production startup requires non-empty {joined}")
    if app_settings.api_key == app_settings.scorer_api_key:
        raise RuntimeError("production API_KEY and SCORER_API_KEY must be distinct")


def create_app(
    app_settings: Settings = settings,
    *,
    database_engine: Engine = engine,
) -> FastAPI:
    validate_security_settings(app_settings)
    configure_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if app_settings.auto_create_schema:
            init_db(database_engine)
            with session_scope(database_engine) as session:
                ensure_default_organization(session)
        yield

    application = FastAPI(
        title="Insider Threat Framework API",
        version="0.1.0",
        summary="Person -> Role -> Global behavioral backoff backend",
        description=(
            "Stores label-free canonical events and user-day artifacts, evaluates "
            "Feature and Sequence readiness independently, fuses calibrated scores, "
            "and manages alerts plus quarantined personal-profile updates."
        ),
        lifespan=lifespan,
        openapi_tags=[
            {"name": "health", "description": "Liveness and database readiness"},
            {"name": "framework", "description": "Locked framework and feature catalog"},
            {"name": "identity", "description": "Users and effective-dated roles"},
            {"name": "ingestion", "description": "Idempotent jobs and checkpoints"},
            {"name": "events", "description": "Label-free canonical event storage"},
            {"name": "user-days", "description": "Feature128 and Sequence7 artifacts"},
            {"name": "artifacts", "description": "Checksummed artifact manifests"},
            {"name": "scoring", "description": "References, readiness, fusion, decisions"},
            {"name": "alerts", "description": "Mutable analyst workflow"},
            {"name": "safe-update", "description": "Quarantined profile-update decisions"},
        ],
    )
    application.state.settings = app_settings
    application.add_middleware(RequestContextMiddleware, settings=app_settings)
    if app_settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(app_settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "PATCH", "OPTIONS"],
            allow_headers=[
                "content-type",
                "x-api-key",
                "x-scorer-api-key",
                "x-request-id",
            ],
        )
    register_exception_handlers(application)
    application.include_router(router)
    return application


app: Any = create_app()
