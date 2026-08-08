from __future__ import annotations

from typing import Any


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def not_found(resource: str, identifier: str) -> ApiError:
    return ApiError(404, "NOT_FOUND", f"{resource} was not found", {"id": identifier})


def conflict(code: str, message: str, details: Any | None = None) -> ApiError:
    return ApiError(409, code, message, details)
