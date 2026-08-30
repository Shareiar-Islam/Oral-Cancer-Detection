"""Shared fixtures.

The tests never load the real checkpoint: a stub model is injected into the
loader singleton so the suite runs fast and deterministically, and so the
sigmoid/softmax branches can both be exercised on demand.
"""

from __future__ import annotations

import io
from typing import Iterator

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn

from app import model_loader
from app.config import get_settings
from app.model_loader import ModelInfo


class ConstantModel(nn.Module):
    """A model that ignores its input and returns a fixed output tensor.

    Fixing the output is what makes assertions about post-processing exact --
    we are testing the sigmoid/softmax/threshold logic, not the weights.
    """

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("output", output)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if x.shape[1:] != (3, 224, 224):
            raise AssertionError(f"unexpected input shape {tuple(x.shape)}")
        self.calls += 1
        return self.output


def make_info(**overrides: object) -> ModelInfo:
    """A ModelInfo matching EXP-4A, with per-test overrides."""
    defaults: dict[str, object] = {
        "checkpoint_path": "/tmp/test_model.pkl",
        "device": "cpu",
        "num_output_units": 1,
        "activation": "sigmoid",
        "outputs_probability": False,
        "threshold": 0.5,
        "positive_class_index": 1,
        "class_names": {0: "Non-Cancer", 1: "Cancer"},
    }
    defaults.update(overrides)
    return ModelInfo(**defaults)  # type: ignore[arg-type]


def make_image_bytes(
    size: tuple[int, int] = (640, 480),
    fmt: str = "JPEG",
    color: tuple[int, int, int] = (180, 90, 90),
    exif_orientation: int | None = None,
) -> bytes:
    """An in-memory test image, optionally tagged with an EXIF orientation."""
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    kwargs: dict[str, object] = {}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[274] = exif_orientation  # 274 == Orientation
        kwargs["exif"] = exif.tobytes()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that opens a real network connection.

    The download tests substitute a fake transport. When the production code
    changed which function performs the request, those substitutions silently
    stopped applying and the suite began contacting the internet instead of
    failing -- tests that pass for the wrong reason, and would break offline or
    in CI. This makes that mistake loud rather than invisible.
    """
    import socket

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "A test attempted a real network connection. Substitute the "
            "transport (see model_loader._urlopen) instead of reaching out."
        )

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture(autouse=True)
def _clear_model_state() -> Iterator[None]:
    """Guarantee no test leaks a loaded model into the next."""
    yield
    model_loader.teardown()


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def stub_model():
    """Install a single-logit stub (the EXP-4A shape) and return (model, info)."""

    def _install(
        output: torch.Tensor | float = 1.9231, **info_overrides: object
    ) -> tuple[ConstantModel, ModelInfo]:
        tensor = (
            output
            if isinstance(output, torch.Tensor)
            else torch.tensor([[float(output)]])
        )
        units = tensor.reshape(-1).numel()
        info_overrides.setdefault("num_output_units", units)
        info = make_info(**info_overrides)
        model = ConstantModel(tensor)
        model.eval()
        model_loader.set_model(model, info)
        return model, info

    return _install


@pytest.fixture
def no_startup_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the lifespan from loading the real checkpoint.

    The suite must pass on a clean checkout with no .pkl present, and reloading
    a 15 MB model per test would dominate the runtime.
    """
    monkeypatch.setattr(model_loader, "initialise", lambda *a, **k: make_info())


@pytest.fixture
def client(stub_model, no_startup_load) -> Iterator[TestClient]:
    """A TestClient with a stub model installed and startup loading disabled.

    raise_server_exceptions=False so handler-produced 500s are asserted as
    responses rather than re-raised into the test.
    """
    from app.main import app

    stub_model()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def bare_client(no_startup_load) -> Iterator[TestClient]:
    """A TestClient with NO model loaded, for 503 assertions."""
    from app.main import app

    model_loader.teardown()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        model_loader.teardown()
        yield test_client
