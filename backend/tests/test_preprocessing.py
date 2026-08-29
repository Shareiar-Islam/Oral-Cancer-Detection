"""Preprocessing: tensor shape/dtype, normalisation, EXIF, format rejection."""

from __future__ import annotations

import io

import pytest
import torch
from PIL import Image

from app.config import get_settings
from app.exceptions import InvalidImageError, UnsupportedMediaTypeError
from app.preprocessing import decode_image, preprocess
from tests.conftest import make_image_bytes


def test_output_shape_and_dtype() -> None:
    tensor, meta = preprocess(make_image_bytes((640, 480)))
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert meta.original_size == (640, 480)
    assert meta.processed_size == (224, 224)


@pytest.mark.parametrize("size", [(50, 50), (4032, 3024), (224, 224), (1, 1)])
def test_any_input_size_normalises_to_224(size: tuple[int, int]) -> None:
    tensor, _ = preprocess(make_image_bytes(size))
    assert tensor.shape == (1, 3, 224, 224)


def test_normalization_is_applied() -> None:
    """A mid-grey image must land at the exact ImageNet-normalised value.

    Pins both that Normalize ran and that it used the recorded constants:
        (0.5 - mean) / std  per channel
    """
    settings = get_settings()
    tensor, _ = preprocess(make_image_bytes((100, 100), color=(128, 128, 128)))
    grey = 128 / 255.0
    for channel, (mean, std) in enumerate(
        zip(settings.normalize_mean, settings.normalize_std)
    ):
        expected = (grey - mean) / std
        assert tensor[0, channel].mean().item() == pytest.approx(expected, abs=1e-3)


def test_normalization_actually_shifts_values() -> None:
    """Guard against Normalize being silently dropped from the Compose."""
    tensor, _ = preprocess(make_image_bytes(color=(0, 0, 0)))
    # A black image post-normalisation is strongly negative, never in [0,1].
    assert tensor.min().item() < -1.0


def test_grayscale_is_converted_to_three_channels() -> None:
    buffer = io.BytesIO()
    Image.new("L", (300, 300), 120).save(buffer, format="PNG")
    tensor, meta = preprocess(buffer.getvalue())
    assert tensor.shape[1] == 3
    assert meta.mode == "L"


def test_rgba_alpha_is_dropped() -> None:
    buffer = io.BytesIO()
    Image.new("RGBA", (200, 200), (10, 20, 30, 128)).save(buffer, format="PNG")
    tensor, meta = preprocess(buffer.getvalue())
    assert tensor.shape[1] == 3
    assert meta.mode == "RGBA"


# --- EXIF ------------------------------------------------------------------
def test_exif_orientation_flag_is_reported() -> None:
    _, meta = preprocess(make_image_bytes((640, 480), exif_orientation=6))
    assert meta.exif_corrected is True


def test_no_exif_means_no_correction() -> None:
    _, meta = preprocess(make_image_bytes((640, 480)))
    assert meta.exif_corrected is False


def test_orientation_1_is_not_treated_as_a_correction() -> None:
    _, meta = preprocess(make_image_bytes((640, 480), exif_orientation=1))
    assert meta.exif_corrected is False


def test_exif_rotation_is_actually_applied_to_pixels() -> None:
    """Orientation 6 means 'rotate 90 CW to display'.

    A landscape image tagged 6 must come out portrait -- proving the pixels
    moved, not just that a flag was reported.
    """
    image = Image.new("RGB", (400, 200), (0, 0, 0))
    # Mark the left half so the rotation is detectable by position.
    for x in range(200):
        for y in range(200):
            image.putpixel((x, y), (255, 255, 255))

    exif = Image.Exif()
    exif[274] = 6
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())

    rotated, meta = decode_image(buffer.getvalue())

    assert meta.exif_corrected is True
    assert meta.original_size == (400, 200)          # as stored
    assert rotated.size == (200, 400)                 # as displayed: portrait
    # The white block was on the left; a 90 degree CW rotation puts it on top.
    assert rotated.getpixel((100, 30))[0] > 200
    assert rotated.getpixel((100, 370))[0] < 50


def test_exif_rotation_survives_into_the_tensor() -> None:
    """The rotation must happen BEFORE resize, so it reaches the model."""
    image = Image.new("RGB", (400, 200), (0, 0, 0))
    for x in range(400):
        for y in range(100):
            image.putpixel((x, y), (255, 255, 255))  # white top half
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 CW for display -> the top edge moves to the RIGHT
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())

    tensor, _ = preprocess(buffer.getvalue())
    left = tensor[0, :, :, :112].mean().item()
    right = tensor[0, :, :, 112:].mean().item()
    assert right > left, "EXIF rotation did not reach the model input"

    # Control: the identical image WITHOUT the EXIF tag must not be rotated,
    # so the bright half stays where it started (top, not right).
    plain = io.BytesIO()
    image.save(plain, format="JPEG")
    untouched, _ = preprocess(plain.getvalue())
    assert untouched[0, :, :112, :].mean() > untouched[0, :, 112:, :].mean()
    assert abs(
        untouched[0, :, :, :112].mean() - untouched[0, :, :, 112:].mean()
    ) < 0.1, "unrotated image should not differ left-to-right"


# --- rejection paths -------------------------------------------------------
def test_empty_bytes_rejected() -> None:
    with pytest.raises(InvalidImageError):
        preprocess(b"")


def test_non_image_bytes_rejected() -> None:
    with pytest.raises(InvalidImageError):
        preprocess(b"this is definitely not an image, it is prose")


def test_truncated_image_rejected() -> None:
    data = make_image_bytes((800, 600))
    with pytest.raises(InvalidImageError):
        preprocess(data[: len(data) // 3])


def test_pdf_disguised_as_image_rejected() -> None:
    """Validation must decode, not trust the extension or content-type."""
    with pytest.raises(InvalidImageError):
        preprocess(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\nfake pdf body")


def test_unsupported_but_decodable_format_rejected() -> None:
    """GIF decodes fine in PIL but is not on the accepted list."""
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100), (1, 2, 3)).save(buffer, format="GIF")
    with pytest.raises(UnsupportedMediaTypeError):
        preprocess(buffer.getvalue())


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "BMP", "TIFF"])
def test_all_accepted_formats_decode(fmt: str) -> None:
    tensor, meta = preprocess(make_image_bytes((320, 240), fmt=fmt))
    assert tensor.shape == (1, 3, 224, 224)
    assert meta.pil_format == fmt


def test_transform_is_a_module_level_singleton() -> None:
    """The Compose must be built once, not per request."""
    from app import preprocessing

    first = preprocessing.TRANSFORM
    preprocess(make_image_bytes())
    preprocess(make_image_bytes())
    assert preprocessing.TRANSFORM is first
