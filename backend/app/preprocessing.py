"""Image bytes -> normalised model input tensor.

Pipeline (exact order, matching EXP-4A's recorded training setup):

    bytes -> PIL.Image.open
          -> ImageOps.exif_transpose      # honour camera orientation
          -> convert("RGB")               # drop alpha / expand greyscale
          -> Resize((224, 224))           # see RESIZE_MODE note below
          -> ToTensor()                   # [0,1], CHW
          -> Normalize(ImageNet mean/std)
          -> unsqueeze(0)                 # (1, 3, 224, 224)

The Compose is built ONCE at import and reused for every request.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Callable

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from .config import get_settings
from .exceptions import InvalidImageError, UnsupportedMediaTypeError

logger = logging.getLogger(__name__)

# PIL format names we accept, checked against what PIL actually decoded --
# never against the client-supplied Content-Type, which is trivially spoofed.
ALLOWED_PIL_FORMATS: frozenset[str] = frozenset(
    {"JPEG", "JPEG2000", "PNG", "WEBP", "BMP", "TIFF", "MPO"}
)

# Corresponding MIME types, advertised via /api/model-info for the frontend.
ALLOWED_MIME_TYPES: tuple[str, ...] = (
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
)

# EXIF tag 274 == Orientation. Value 1 (or absent) means "already upright".
_EXIF_ORIENTATION_TAG = 274

_settings = get_settings()


def _build_transform() -> transforms.Compose:
    """Construct the eval-time transform once, from config.

    RESIZE_MODE is the single knob that switches eval geometry. EXP-4A trained
    on a pre-resized 224px dataset, so "resize" (a direct square resize) is the
    faithful default. If your training notebook's *validation* transform was
    `Resize(256) + CenterCrop(224)` instead, set RESIZE_MODE=resize_crop in
    .env -- no code change needed.
    """
    size = _settings.input_size
    if _settings.resize_mode == "resize_crop":
        resize_to = int(round(size / _settings.center_crop_ratio))
        geometry: list[Callable[[Image.Image], Image.Image]] = [
            transforms.Resize(resize_to),
            transforms.CenterCrop(size),
        ]
    else:
        geometry = [transforms.Resize((size, size))]

    return transforms.Compose(
        [
            *geometry,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(_settings.normalize_mean),
                std=list(_settings.normalize_std),
            ),
        ]
    )


# Module-level singleton: built once, never per request.
TRANSFORM: transforms.Compose = _build_transform()


@dataclass(frozen=True)
class ImageMetadata:
    """What we can report about the upload without ever storing it."""

    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    exif_corrected: bool
    pil_format: str
    mode: str


def _needs_exif_rotation(image: Image.Image) -> bool:
    """True when the file carries a non-trivial EXIF orientation."""
    try:
        exif = image.getexif()
    except Exception:  # noqa: BLE001 - malformed EXIF must never fail a request
        return False
    if not exif:
        return False
    return exif.get(_EXIF_ORIENTATION_TAG, 1) not in (1, None)


def decode_image(data: bytes) -> tuple[Image.Image, ImageMetadata]:
    """Decode bytes into an upright RGB image plus reportable metadata.

    Validation is done by *decoding*, not by trusting any declared type.
    """
    if not data:
        raise InvalidImageError("The uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # force a full decode so truncated files fail here
    except UnidentifiedImageError as exc:
        raise InvalidImageError(
            "The file could not be read as an image. It may be corrupt or "
            "not an image at all."
        ) from exc
    except OSError as exc:
        raise InvalidImageError(
            f"The image is damaged or incomplete and could not be decoded ({exc})."
        ) from exc

    fmt = (image.format or "UNKNOWN").upper()
    if fmt not in ALLOWED_PIL_FORMATS:
        raise UnsupportedMediaTypeError(
            f"{fmt} images are not supported. Please upload one of: "
            f"{', '.join(sorted(ALLOWED_PIL_FORMATS - {'MPO', 'JPEG2000'}))}."
        )

    original_size = (image.width, image.height)
    original_mode = image.mode
    exif_corrected = _needs_exif_rotation(image)

    # Orientation first: rotating after the resize would distort the crop.
    upright = ImageOps.exif_transpose(image) or image
    rgb = upright.convert("RGB")

    metadata = ImageMetadata(
        original_size=original_size,
        processed_size=(_settings.input_size, _settings.input_size),
        exif_corrected=exif_corrected,
        pil_format=fmt,
        mode=original_mode,
    )
    return rgb, metadata


def preprocess(data: bytes) -> tuple[torch.Tensor, ImageMetadata]:
    """Full path from raw upload bytes to a (1, 3, H, W) input batch."""
    image, metadata = decode_image(data)
    tensor: torch.Tensor = TRANSFORM(image).unsqueeze(0)
    return tensor, metadata
