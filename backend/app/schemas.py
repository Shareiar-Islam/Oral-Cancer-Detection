"""Pydantic v2 request/response models -- the API's public contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = Field(description="'degraded' when the model failed to load.")
    model_loaded: bool
    device: str

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {"status": "ok", "model_loaded": True, "device": "cpu"}
        },
    )


class ModelInfoResponse(BaseModel):
    """What the server actually loaded -- use this to audit the deployment."""

    architecture: str
    checkpoint: str = Field(description="Checkpoint filename (path is not exposed).")
    device: str
    input_size: list[int] = Field(description="[height, width] fed to the model.")
    num_output_units: int = Field(description="1 => sigmoid head, 2 => softmax head.")
    activation: str
    outputs_probability: bool
    threshold: float
    positive_class_index: int
    class_names: dict[int, str]
    normalization: dict[str, list[float]]
    resize_mode: str
    accepted_mime_types: list[str]
    max_upload_mb: float
    max_batch_files: int
    load_path: str = Field(description="How the checkpoint was read and reconstructed.")
    missing_keys: int
    unexpected_keys: int
    checkpoint_metadata: dict = Field(
        default_factory=dict,
        description="Non-tensor bookkeeping recorded by the training script.",
    )

    model_config = ConfigDict(protected_namespaces=())


class ImageInfo(BaseModel):
    original_size: list[int] = Field(description="[width, height] as uploaded.")
    processed_size: list[int] = Field(description="[width, height] seen by the model.")
    exif_corrected: bool = Field(description="True when EXIF orientation was applied.")
    format: str
    mode: str


class PredictionResponse(BaseModel):
    prediction: str = Field(description="The predicted class name.")
    probability: float = Field(ge=0.0, le=1.0, description="P(Cancer), independent of the prediction.")
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the predicted class.")
    threshold: float
    raw_output: float = Field(description="Pre-activation model output (the logit).")
    inference_time_ms: float
    image: ImageInfo

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": "Cancer",
                "probability": 0.8734,
                "confidence": 0.8734,
                "threshold": 0.5,
                "raw_output": 1.9231,
                "inference_time_ms": 42.7,
                "image": {
                    "original_size": [3024, 4032],
                    "processed_size": [224, 224],
                    "exif_corrected": True,
                    "format": "JPEG",
                    "mode": "RGB",
                },
            }
        }
    )


class BatchItemError(BaseModel):
    code: str
    message: str


class BatchItem(BaseModel):
    """One entry per uploaded file. Exactly one of result/error is set."""

    filename: str
    success: bool
    result: PredictionResponse | None = None
    error: BatchItemError | None = None


class BatchPredictionResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[BatchItem]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    """The shape every failure returns."""

    error: ErrorDetail

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": "INVALID_IMAGE",
                    "message": "The file could not be read as an image.",
                }
            }
        }
    )
