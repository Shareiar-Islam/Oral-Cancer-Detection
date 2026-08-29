"""API contract: happy path, validation, error shape, batch behaviour."""

from __future__ import annotations

import io
import math

import pytest
import torch
from PIL import Image

from app.config import get_settings
from tests.conftest import make_image_bytes


def upload(data: bytes, name: str = "lesion.jpg", mime: str = "image/jpeg"):
    return {"file": (name, data, mime)}


# --- health / model-info ---------------------------------------------------
def test_health_reports_ok_when_loaded(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "device": "cpu"}


def test_health_reports_degraded_without_a_model(bare_client) -> None:
    """Health must stay 200 so the UI can tell 'offline' from 'model broken'."""
    response = bare_client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_model_info_describes_the_loaded_model(client) -> None:
    body = client.get("/api/model-info").json()
    assert body["architecture"] == "torchvision.models.efficientnet_b0"
    assert body["num_output_units"] == 1
    assert body["activation"] == "sigmoid"
    assert body["threshold"] == 0.5
    assert body["positive_class_index"] == 1
    assert body["class_names"] == {"0": "Non-Cancer", "1": "Cancer"}
    assert body["input_size"] == [224, 224]
    assert body["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert body["normalization"]["std"] == [0.229, 0.224, 0.225]
    assert "image/jpeg" in body["accepted_mime_types"]


def test_model_info_does_not_leak_the_filesystem_path(client) -> None:
    assert "/" not in client.get("/api/model-info").json()["checkpoint"]


# --- predict: happy path ---------------------------------------------------
def test_predict_returns_the_documented_shape(client) -> None:
    response = client.post("/api/predict", files=upload(make_image_bytes((3024, 4032))))
    assert response.status_code == 200
    body = response.json()

    assert set(body) == {
        "prediction", "probability", "confidence", "threshold",
        "raw_output", "inference_time_ms", "image",
    }
    assert set(body["image"]) == {
        "original_size", "processed_size", "exif_corrected", "format", "mode",
    }
    assert body["image"]["original_size"] == [3024, 4032]
    assert body["image"]["processed_size"] == [224, 224]
    assert body["inference_time_ms"] >= 0


def test_predict_applies_sigmoid_to_the_logit(client, stub_model) -> None:
    stub_model(1.9231)
    body = client.post("/api/predict", files=upload(make_image_bytes())).json()
    assert body["raw_output"] == pytest.approx(1.9231, abs=1e-4)
    assert body["probability"] == pytest.approx(1 / (1 + math.exp(-1.9231)), abs=1e-5)
    assert body["prediction"] == "Cancer"
    assert body["confidence"] == pytest.approx(body["probability"], abs=1e-6)


def test_predict_non_cancer_path(client, stub_model) -> None:
    stub_model(-2.5)
    body = client.post("/api/predict", files=upload(make_image_bytes())).json()
    assert body["prediction"] == "Non-Cancer"
    assert body["probability"] < 0.5
    assert body["confidence"] == pytest.approx(1 - body["probability"], abs=1e-6)


def test_predict_reports_exif_correction(client) -> None:
    body = client.post(
        "/api/predict", files=upload(make_image_bytes(exif_orientation=6))
    ).json()
    assert body["image"]["exif_corrected"] is True


def test_model_actually_received_a_224_batch(client, stub_model) -> None:
    """ConstantModel asserts its input shape, so reaching 200 proves it."""
    model, _ = stub_model(0.5)
    client.post("/api/predict", files=upload(make_image_bytes((800, 600))))
    assert model.calls == 1


@pytest.mark.parametrize(
    ("fmt", "mime"),
    [("PNG", "image/png"), ("WEBP", "image/webp"), ("BMP", "image/bmp"), ("TIFF", "image/tiff")],
)
def test_predict_accepts_every_documented_format(client, fmt: str, mime: str) -> None:
    response = client.post(
        "/api/predict", files=upload(make_image_bytes(fmt=fmt), f"x.{fmt.lower()}", mime)
    )
    assert response.status_code == 200


def test_content_type_is_not_trusted(client) -> None:
    """A real PNG mislabelled as JPEG must still work -- PIL decides, not the header."""
    response = client.post(
        "/api/predict", files=upload(make_image_bytes(fmt="PNG"), "lie.jpg", "image/jpeg")
    )
    assert response.status_code == 200
    assert response.json()["image"]["format"] == "PNG"


def test_lying_content_type_cannot_smuggle_a_non_image(client) -> None:
    response = client.post(
        "/api/predict", files=upload(b"#!/bin/sh\nrm -rf /", "evil.jpg", "image/jpeg")
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


# --- predict: rejection ----------------------------------------------------
def test_oversized_upload_returns_413(client) -> None:
    settings = get_settings()
    oversized = b"\xff\xd8\xff\xe0" + b"\x00" * (settings.max_upload_bytes + 1024)
    response = client.post("/api/predict", files=upload(oversized, "huge.jpg"))
    assert response.status_code == 413
    body = response.json()
    assert body["error"]["code"] == "FILE_TOO_LARGE"
    assert "10" in body["error"]["message"]


def test_large_but_allowed_upload_is_accepted(client) -> None:
    """A genuinely big photo under the cap must not be rejected."""
    data = make_image_bytes((3000, 2000))
    assert len(data) < get_settings().max_upload_bytes
    assert client.post("/api/predict", files=upload(data)).status_code == 200


def test_corrupt_image_returns_422(client) -> None:
    data = make_image_bytes((800, 600))
    response = client.post("/api/predict", files=upload(data[: len(data) // 3]))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_non_image_returns_422(client) -> None:
    response = client.post("/api/predict", files=upload(b"just some text", "notes.txt", "text/plain"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_unsupported_format_returns_415(client) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100), (1, 2, 3)).save(buffer, format="GIF")
    response = client.post("/api/predict", files=upload(buffer.getvalue(), "a.gif", "image/gif"))
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_empty_file_returns_422(client) -> None:
    response = client.post("/api/predict", files=upload(b"", "empty.jpg"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "EMPTY_UPLOAD"


def test_missing_file_field_returns_422(client) -> None:
    response = client.post("/api/predict")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_wrong_field_name_returns_422(client) -> None:
    response = client.post("/api/predict", files={"image": ("a.jpg", make_image_bytes(), "image/jpeg")})
    assert response.status_code == 422


def test_predict_without_a_model_returns_503(bare_client) -> None:
    response = bare_client.post("/api/predict", files=upload(make_image_bytes()))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_NOT_LOADED"


def test_unknown_route_uses_the_standard_error_shape(client) -> None:
    body = client.get("/api/nope").json()
    assert body["error"]["code"] == "NOT_FOUND"


def test_every_error_has_the_same_envelope(client) -> None:
    """One shape for every failure, so the frontend needs one parser."""
    failures = [
        client.post("/api/predict", files=upload(b"nope", "a.jpg")),
        client.post("/api/predict", files=upload(b"", "b.jpg")),
        client.post("/api/predict"),
        client.get("/api/nope"),
    ]
    for response in failures:
        body = response.json()
        assert set(body) == {"error"}
        assert set(body["error"]) == {"code", "message"}
        assert isinstance(body["error"]["message"], str) and body["error"]["message"]


def test_request_id_header_is_returned(client) -> None:
    response = client.post("/api/predict", files=upload(make_image_bytes()))
    assert len(response.headers.get("X-Request-ID", "")) == 12


# --- batch -----------------------------------------------------------------
def test_batch_happy_path(client) -> None:
    files = [("files", (f"a{i}.jpg", make_image_bytes(), "image/jpeg")) for i in range(3)]
    body = client.post("/api/predict/batch", files=files).json()
    assert body["total"] == 3
    assert body["succeeded"] == 3
    assert body["failed"] == 0
    assert all(item["success"] and item["result"] for item in body["results"])


def test_batch_isolates_a_bad_file(client) -> None:
    """One corrupt file must not sink the whole batch."""
    files = [
        ("files", ("good.jpg", make_image_bytes(), "image/jpeg")),
        ("files", ("bad.jpg", b"not an image", "image/jpeg")),
        ("files", ("good2.png", make_image_bytes(fmt="PNG"), "image/png")),
    ]
    body = client.post("/api/predict/batch", files=files).json()

    assert (body["total"], body["succeeded"], body["failed"]) == (3, 2, 1)
    good, bad, good2 = body["results"]
    assert good["success"] and good["result"]["prediction"]
    assert bad["success"] is False
    assert bad["error"]["code"] == "INVALID_IMAGE"
    assert bad["result"] is None
    assert good2["success"] is True


def test_batch_preserves_filenames_and_order(client) -> None:
    names = ["first.jpg", "second.jpg", "third.jpg"]
    files = [("files", (n, make_image_bytes(), "image/jpeg")) for n in names]
    body = client.post("/api/predict/batch", files=files).json()
    assert [item["filename"] for item in body["results"]] == names


def test_batch_over_the_limit_returns_413(client) -> None:
    limit = get_settings().max_batch_files
    files = [("files", (f"{i}.jpg", make_image_bytes(), "image/jpeg")) for i in range(limit + 1)]
    response = client.post("/api/predict/batch", files=files)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "TOO_MANY_FILES"


def test_batch_at_exactly_the_limit_is_accepted(client) -> None:
    limit = get_settings().max_batch_files
    files = [("files", (f"{i}.jpg", make_image_bytes((64, 64)), "image/jpeg")) for i in range(limit)]
    body = client.post("/api/predict/batch", files=files).json()
    assert body["total"] == limit and body["succeeded"] == limit


def test_batch_without_a_model_returns_503(bare_client) -> None:
    files = [("files", ("a.jpg", make_image_bytes(), "image/jpeg"))]
    assert bare_client.post("/api/predict/batch", files=files).status_code == 503


# --- CORS ------------------------------------------------------------------
def test_cors_allows_the_configured_origin(client) -> None:
    response = client.options(
        "/api/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
