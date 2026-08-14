from __future__ import annotations

import logging
import re
import secrets
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.errors import ApiError
from app.validation import DomainValidationError

logger = logging.getLogger("insider_threat.api")
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:@-]{1,160}$")


def require_scorer_api_key(request: Request) -> None:
    """Authorize the trusted scoring worker and bind its server-side principal.

    Production always configures a dedicated scorer key. Development may omit
    it; in that case the normal API key (when enabled) remains the credential.
    """

    runtime_settings = request.app.state.settings
    scorer_api_key = getattr(runtime_settings, "scorer_api_key", None)
    if not scorer_api_key:
        return

    supplied_key = request.headers.get("x-scorer-api-key")
    if supplied_key is None or not secrets.compare_digest(supplied_key, scorer_api_key):
        raise ApiError(
            401,
            "SCORER_UNAUTHORIZED",
            "invalid or missing scorer API key",
        )
    request.state.actor = runtime_settings.scorer_api_key_actor


def error_payload(
    request: Request,
    code: str,
    message: str,
    details: Any | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "request_id": getattr(request.state, "request_id", "unknown"),
        "details": details,
    }


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, settings: Any) -> None:
        self.app = app
        self.settings = settings

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        supplied_request_id = request.headers.get("x-request-id")
        request_id = supplied_request_id.strip() if supplied_request_id else str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.actor = self.settings.api_key_actor
        response_status: int | None = None

        async def send_with_security_headers(message: Message) -> None:
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = int(message["status"])
                headers = MutableHeaders(scope=message)
                headers["x-request-id"] = request_id
                headers["x-content-type-options"] = "nosniff"
                headers["x-frame-options"] = "DENY"
                headers["referrer-policy"] = "no-referrer"
            await send(message)

        if supplied_request_id is not None and not REQUEST_ID_PATTERN.fullmatch(request_id):
            request_id = "invalid"
            request.state.request_id = "invalid"
            response = JSONResponse(
                status_code=400,
                content=error_payload(
                    request,
                    "INVALID_REQUEST_ID",
                    "x-request-id must contain 1-160 safe characters",
                ),
            )
            await response(scope, receive, send_with_security_headers)
            return

        content_length = request.headers.get("content-length")
        max_bytes = int(getattr(self.settings, "max_request_bytes", 2_097_152))
        if content_length:
            try:
                declared_bytes = int(content_length)
            except ValueError:
                response = JSONResponse(
                    status_code=400,
                    content=error_payload(
                        request, "INVALID_CONTENT_LENGTH", "invalid Content-Length"
                    ),
                )
                await response(scope, receive, send_with_security_headers)
                return
            if declared_bytes < 0:
                response = JSONResponse(
                    status_code=400,
                    content=error_payload(
                        request, "INVALID_CONTENT_LENGTH", "invalid Content-Length"
                    ),
                )
                await response(scope, receive, send_with_security_headers)
                return
            if declared_bytes > max_bytes:
                response = JSONResponse(
                    status_code=413,
                    content=error_payload(
                        request,
                        "REQUEST_TOO_LARGE",
                        f"request body exceeds {max_bytes} bytes",
                    ),
                )
                await response(scope, receive, send_with_security_headers)
                return

        api_key = getattr(self.settings, "api_key", None)
        public_paths = {
            "/health/live",
            "/health/ready",
            "/docs",
            "/docs/oauth2-redirect",
            "/openapi.json",
            "/redoc",
        }
        if api_key and request.url.path not in public_paths:
            supplied_key = request.headers.get("x-api-key")
            if supplied_key is None or not secrets.compare_digest(supplied_key, api_key):
                response = JSONResponse(
                    status_code=401,
                    content=error_payload(request, "UNAUTHORIZED", "invalid or missing API key"),
                )
                await response(scope, receive, send_with_security_headers)
                return

        buffered_messages: list[Message] = []
        received_bytes = 0
        while True:
            message = await receive()
            buffered_messages.append(message)
            if message["type"] == "http.disconnect":
                break
            if message["type"] != "http.request":
                continue
            received_bytes += len(message.get("body", b""))
            if received_bytes > max_bytes:
                response = JSONResponse(
                    status_code=413,
                    content=error_payload(
                        request,
                        "REQUEST_TOO_LARGE",
                        f"request body exceeds {max_bytes} bytes",
                    ),
                )
                await response(scope, receive, send_with_security_headers)
                return
            if not message.get("more_body", False):
                break

        message_index = 0

        async def replay_receive() -> Message:
            nonlocal message_index
            if message_index < len(buffered_messages):
                message = buffered_messages[message_index]
                message_index += 1
                return message
            return {"type": "http.request", "body": b"", "more_body": False}

        started = time.perf_counter()
        try:
            await self.app(scope, replay_receive, send_with_security_headers)
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info(
                "request_complete method=%s path=%s status=%s elapsed_ms=%s request_id=%s",
                request.method,
                request.url.path,
                response_status or 500,
                elapsed_ms,
                request_id,
            )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(request, exc.code, exc.message, exc.details),
        )

    @app.exception_handler(DomainValidationError)
    async def domain_error_handler(request: Request, exc: DomainValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_payload(request, exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors: list[dict[str, Any]] = []
        for item in exc.errors():
            normalized = dict(item)
            if "ctx" in normalized:
                normalized["ctx"] = {key: str(value) for key, value in normalized["ctx"].items()}
            errors.append(normalized)
        return JSONResponse(
            status_code=422,
            content=error_payload(
                request,
                "REQUEST_VALIDATION_ERROR",
                "request validation failed",
                errors,
            ),
        )

    @app.exception_handler(IntegrityError)
    async def integrity_error_handler(request: Request, exc: IntegrityError) -> JSONResponse:
        logger.warning(
            "database_integrity_error request_id=%s error=%s",
            getattr(request.state, "request_id", "unknown"),
            type(exc.orig).__name__,
        )
        return JSONResponse(
            status_code=409,
            content=error_payload(
                request,
                "DATABASE_CONFLICT",
                "the operation conflicts with an existing record or constraint",
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "unhandled_error request_id=%s",
            getattr(request.state, "request_id", "unknown"),
        )
        return JSONResponse(
            status_code=500,
            content=error_payload(request, "INTERNAL_ERROR", "internal server error"),
        )
