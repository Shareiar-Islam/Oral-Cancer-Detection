"""Custom errors and the handlers that render them.

Every failure leaves the API in one consistent JSON shape:

    {"error": {"code": "SNAKE_CASE_CODE", "message": "human readable"}}
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base for errors that map cleanly onto an HTTP response."""

    status_code: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ModelNotLoadedError(AppError):
    """The model failed to load at startup, or is still loading."""

    status_code = 503
    code = "MODEL_NOT_LOADED"


class ModelLoadError(AppError):
    """The checkpoint exists but could not be turned into a usable model."""

    status_code = 503
    code = "MODEL_LOAD_FAILED"


class UnsupportedMediaTypeError(AppError):
    """The upload is not an image format we accept."""

    status_code = 415
    code = "UNSUPPORTED_MEDIA_TYPE"


class InvalidImageError(AppError):
    """The bytes are not a decodable image (truncated, corrupt, mislabelled)."""

    status_code = 422
    code = "INVALID_IMAGE"


class FileTooLargeError(AppError):
    """The upload exceeds MAX_UPLOAD_MB."""

    status_code = 413
    code = "FILE_TOO_LARGE"


class TooManyFilesError(AppError):
    """A batch request exceeded MAX_BATCH_FILES."""

    status_code = 413
    code = "TOO_MANY_FILES"


class EmptyUploadError(AppError):
    """A file field was present but contained zero bytes."""

    status_code = 422
    code = "EMPTY_UPLOAD"


class InferenceError(AppError):
    """The forward pass or post-processing failed."""

    status_code = 500
    code = "INFERENCE_FAILED"


def error_body(code: str, message: str) -> dict[str, dict[str, str]]:
    """The single canonical error payload shape."""
    return {"error": {"code": code, "message": message}}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers so *every* error path returns `error_body`."""

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        # Client mistakes are noise at WARNING; genuine 5xx deserve a trace.
        if exc.status_code >= 500:
            logger.exception("%s: %s", exc.code, exc.message)
        else:
            logger.warning("%s: %s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.code, exc.message),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        detail = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in detail.get("loc", ()) if p != "body")
        message = detail.get("msg", "Request validation failed.")
        if field:
            message = f"{field}: {message}"
        return JSONResponse(
            status_code=422, content=error_body("VALIDATION_ERROR", message)
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        codes = {
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            413: "FILE_TOO_LARGE",
            415: "UNSUPPORTED_MEDIA_TYPE",
        }
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(
                codes.get(exc.status_code, f"HTTP_{exc.status_code}"),
                str(exc.detail),
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Never leak internals to the client; the trace goes to the log only.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=error_body(
                "INTERNAL_ERROR", "An unexpected server error occurred."
            ),
        )
