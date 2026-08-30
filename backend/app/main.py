"""FastAPI application: lifespan, CORS, logging, routes.

Privacy note: uploaded images are held in memory for the duration of a request
and never written to disk, never logged, and never cached. These may be
patient photographs. Only image *dimensions* and the prediction are logged.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import model_loader
from .config import get_settings
from .exceptions import (
    AppError,
    EmptyUploadError,
    FileTooLargeError,
    TooManyFilesError,
    register_exception_handlers,
)
from .inference import run_inference
from .preprocessing import ALLOWED_MIME_TYPES, preprocess
from .schemas import (
    BatchItem,
    BatchItemError,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ImageInfo,
    ModelInfoResponse,
    PredictionResponse,
)

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup; drop it at shutdown.

    A load failure does NOT crash the process -- the app starts in a degraded
    state so /api/health can report the problem instead of the container just
    restart-looping with the traceback buried in the logs.
    """
    try:
        info = model_loader.initialise(settings)
        logger.info("Startup complete. Serving %s on %s.", info.architecture, info.device)
    except Exception:
        logger.exception(
            "MODEL FAILED TO LOAD -- /api/predict will return 503. "
            "Run `python scripts/inspect_model.py <path>` to diagnose."
        )
    yield
    model_loader.teardown()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Oral Cancer Classifier API",
    description=(
        "Binary Cancer / Non-Cancer classification of intraoral photographs "
        "using EfficientNet-B0.\n\n"
        "**Research prototype. Not a diagnostic device.** Output must not be "
        "used for clinical decisions or to replace evaluation by a qualified "
        "clinician."
    ),
    version="1.0.0",
    lifespan=lifespan,
    responses={
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_methods=["GET", "HEAD", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
    **settings.cors_kwargs,  # allow_origins, plus allow_origin_regex if set
)

register_exception_handlers(app)


@app.middleware("http")
async def request_context(request: Request, call_next) -> Response:
    """Tag every request with an id and log its latency."""
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Request-ID"] = request_id
    if request.url.path.startswith("/api/predict"):
        logger.info(
            "rid=%s %s %s -> %d in %.1f ms",
            request_id, request.method, request.url.path,
            response.status_code, elapsed_ms,
        )
    return response


# --------------------------------------------------------------------------
# Upload handling
# --------------------------------------------------------------------------
async def _read_upload(upload: UploadFile) -> bytes:
    """Read an upload, enforcing the size cap without buffering past it.

    Streamed in chunks so an oversized file is rejected before it can be held
    in memory in full.
    """
    limit = settings.max_upload_bytes
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise FileTooLargeError(
                f"'{upload.filename or 'file'}' exceeds the "
                f"{settings.max_upload_mb:g} MB limit."
            )
        chunks.append(chunk)

    if total == 0:
        raise EmptyUploadError(f"'{upload.filename or 'file'}' is empty.")
    return b"".join(chunks)


async def _predict_bytes(data: bytes, filename: str, request_id: str) -> PredictionResponse:
    """Preprocess -> infer -> response. Never persists the image."""
    model = model_loader.get_model()
    info = model_loader.get_info()

    batch, image_meta = preprocess(data)
    result = await run_inference(model, info, batch)

    logger.info(
        "rid=%s file=%r %dx%d exif=%s -> %s p=%.4f raw=%.4f in %.1f ms",
        request_id, filename,
        image_meta.original_size[0], image_meta.original_size[1],
        image_meta.exif_corrected,
        result.prediction, result.probability, result.raw_output,
        result.inference_time_ms,
    )

    return PredictionResponse(
        prediction=result.prediction,
        probability=round(result.probability, 6),
        confidence=round(result.confidence, 6),
        threshold=result.threshold,
        raw_output=round(result.raw_output, 6),
        inference_time_ms=result.inference_time_ms,
        image=ImageInfo(
            original_size=list(image_meta.original_size),
            processed_size=list(image_meta.processed_size),
            exif_corrected=image_meta.exif_corrected,
            format=image_meta.pil_format,
            mode=image_meta.mode,
        ),
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.api_route(
    "/api/health",
    methods=["GET", "HEAD"],
    response_model=HealthResponse,
    tags=["system"],
)
async def health() -> HealthResponse:
    """Liveness + model readiness. Always 200 so the frontend can distinguish
    'backend down' (network error) from 'backend up, model broken'.

    HEAD is registered explicitly. Starlette derives HEAD from GET, but
    FastAPI's APIRoute does not -- and uptime monitors (UptimeRobot among
    them) probe with HEAD by default, so a GET-only route answers the
    keep-warm ping with 405 and the instance is left to spin down anyway.
    """
    loaded = model_loader.is_loaded()
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        device=str(model_loader.get_info().device) if loaded else str(settings.resolved_device),
    )


@app.get("/api/model-info", response_model=ModelInfoResponse, tags=["system"])
async def model_info() -> ModelInfoResponse:
    """Everything the server discovered about the loaded checkpoint."""
    info = model_loader.get_info()
    return ModelInfoResponse(
        architecture=info.architecture,
        checkpoint=info.checkpoint_path.rsplit("/", 1)[-1],
        device=info.device,
        input_size=[info.input_size, info.input_size],
        num_output_units=info.num_output_units,
        activation=info.activation,
        outputs_probability=info.outputs_probability,
        threshold=info.threshold,
        positive_class_index=info.positive_class_index,
        class_names=info.class_names,
        normalization={
            "mean": list(info.normalize_mean),
            "std": list(info.normalize_std),
        },
        resize_mode=info.resize_mode,
        accepted_mime_types=list(ALLOWED_MIME_TYPES),
        max_upload_mb=settings.max_upload_mb,
        max_batch_files=settings.max_batch_files,
        load_path=info.load_path,
        missing_keys=len(info.missing_keys),
        unexpected_keys=len(info.unexpected_keys),
        checkpoint_metadata=info.checkpoint_metadata,
    )


@app.post("/api/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    """Classify a single intraoral photograph."""
    data = await _read_upload(file)
    return await _predict_bytes(
        data, file.filename or "upload", getattr(request.state, "request_id", "-")
    )


@app.post("/api/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
async def predict_batch(
    request: Request, files: list[UploadFile] = File(...)
) -> BatchPredictionResponse:
    """Classify up to MAX_BATCH_FILES images.

    A failure on one file yields an error entry for that file only -- the rest
    of the batch still returns results.
    """
    if len(files) > settings.max_batch_files:
        raise TooManyFilesError(
            f"{len(files)} files submitted; the maximum is {settings.max_batch_files}."
        )

    # Fail fast on a hard dependency rather than repeating it per file.
    model_loader.get_model()
    request_id = getattr(request.state, "request_id", "-")

    items: list[BatchItem] = []
    for upload in files:
        name = upload.filename or "upload"
        try:
            data = await _read_upload(upload)
            result = await _predict_bytes(data, name, request_id)
            items.append(BatchItem(filename=name, success=True, result=result))
        except AppError as exc:
            items.append(
                BatchItem(
                    filename=name,
                    success=False,
                    error=BatchItemError(code=exc.code, message=exc.message),
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not sink the batch
            logger.exception("rid=%s batch item %r failed", request_id, name)
            items.append(
                BatchItem(
                    filename=name,
                    success=False,
                    error=BatchItemError(
                        code="INTERNAL_ERROR",
                        message=f"Unexpected failure processing this file ({type(exc).__name__}).",
                    ),
                )
            )

    succeeded = sum(1 for i in items if i.success)
    return BatchPredictionResponse(
        total=len(items), succeeded=succeeded,
        failed=len(items) - succeeded, results=items,
    )
