"""Checkpoint loading: format fallbacks, prefix stripping, head inference.

The synthetic checkpoints here are tiny stand-ins -- a two-layer stub rather
than a real EfficientNet -- so the suite stays fast. The one test that touches
the genuine EXP-4A file skips automatically when it is absent.
"""

from __future__ import annotations

import pickle
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from torch import nn

from app.config import get_settings
from app.exceptions import ModelLoadError
from app.model_loader import (
    _infer_num_outputs,
    _parse_class_names,
    _strip_prefix,
    _titlecase,
    _unwrap_state_dict,
    load_model,
)


def fake_state(num_outputs: int = 1, prefix: str = "") -> dict[str, torch.Tensor]:
    return {
        f"{prefix}features.0.0.weight": torch.randn(32, 3, 3, 3),
        f"{prefix}classifier.1.weight": torch.randn(num_outputs, 1280),
        f"{prefix}classifier.1.bias": torch.randn(num_outputs),
    }


# --- locating the weights --------------------------------------------------
def test_flat_state_dict_is_found() -> None:
    state, where = _unwrap_state_dict(fake_state())
    assert state is not None and where == "top-level"


@pytest.mark.parametrize("key", ["state_dict", "model_state_dict", "model", "net", "weights"])
def test_nested_state_dict_is_found(key: str) -> None:
    state, where = _unwrap_state_dict({key: fake_state(), "epoch": 11})
    assert state is not None and where == key


def test_state_dict_recovered_from_an_unusual_nesting() -> None:
    """Falls back to a recursive search when the key is not a known one."""
    state, where = _unwrap_state_dict({"checkpoint": {"weights_v2": fake_state()}})
    assert state is not None and where == "recovered by search"


def test_live_module_yields_its_state_dict() -> None:
    state, where = _unwrap_state_dict(nn.Linear(4, 2))
    assert state is not None and where == "nn.Module"


def test_metadata_only_dict_yields_nothing() -> None:
    state, _ = _unwrap_state_dict({"epoch": 11, "notes": "no weights here"})
    assert state is None


# --- prefixes --------------------------------------------------------------
@pytest.mark.parametrize("prefix", ["module.", "model.", "_orig_mod."])
def test_uniform_prefix_is_stripped(prefix: str) -> None:
    stripped, found = _strip_prefix(fake_state(prefix=prefix))
    assert found == prefix
    assert all(not k.startswith(prefix) for k in stripped)
    assert "classifier.1.weight" in stripped


def test_unprefixed_keys_are_left_alone() -> None:
    original = fake_state()
    stripped, found = _strip_prefix(original)
    assert found is None and stripped == original


def test_partial_prefix_is_not_stripped() -> None:
    """Only a prefix on EVERY key is a wrapper; a partial match is real structure."""
    mixed = {"module.a.weight": torch.randn(2, 2), "b.weight": torch.randn(2, 2)}
    _, found = _strip_prefix(mixed)
    assert found is None


# --- head inference --------------------------------------------------------
@pytest.mark.parametrize("units", [1, 2])
def test_output_units_read_from_the_head(units: int) -> None:
    assert _infer_num_outputs(fake_state(units)) == units


def test_head_is_preferred_over_earlier_2d_weights() -> None:
    state = OrderedDict(
        [
            ("classifier.1.weight", torch.randn(1, 1280)),
            ("aux.weight", torch.randn(512, 512)),  # later, but not a head
        ]
    )
    assert _infer_num_outputs(state) == 1


def test_fc_style_head_is_recognised() -> None:
    assert _infer_num_outputs({"fc.weight": torch.randn(2, 512)}) == 2


def test_no_classifier_weight_raises_a_useful_error() -> None:
    with pytest.raises(ModelLoadError, match="inspect_model"):
        _infer_num_outputs({"features.0.weight": torch.randn(8, 3, 3, 3)})


# --- class mapping ---------------------------------------------------------
def test_index_to_name_mapping() -> None:
    assert _parse_class_names({"class_names": {0: "NON CANCER", 1: "CANCER"}}) == {
        0: "NON CANCER", 1: "CANCER"
    }


def test_class_to_idx_mapping_is_inverted_correctly() -> None:
    """torchvision's ImageFolder records name -> index; we need index -> name."""
    assert _parse_class_names({"class_to_idx": {"non_cancer": 0, "cancer": 1}}) == {
        0: "non_cancer", 1: "cancer"
    }


def test_ordered_list_of_names() -> None:
    assert _parse_class_names({"classes": ["non cancer", "cancer"]}) == {
        0: "non cancer", 1: "cancer"
    }


def test_missing_mapping_returns_none() -> None:
    assert _parse_class_names({"epoch": 11}) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("NON CANCER", "Non-Cancer"),
        ("CANCER", "Cancer"),
        ("non_cancer", "Non-Cancer"),
        ("Non-Cancer", "Non-Cancer"),
        ("malignant", "Malignant"),
    ],
)
def test_label_titlecasing(raw: str, expected: str) -> None:
    assert _titlecase(raw) == expected


# --- end-to-end load -------------------------------------------------------
def _write_efficientnet_checkpoint(path: Path, *, units: int, use_torch_save: bool,
                                   prefix: str = "", nest: str | None = "state_dict") -> None:
    from torchvision import models

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, units)
    state = {f"{prefix}{k}": v for k, v in model.state_dict().items()}
    payload: object = {nest: state, "class_names": {0: "NON CANCER", 1: "CANCER"}} if nest else state

    if use_torch_save:
        torch.save(payload, path)
    else:
        with path.open("wb") as fh:
            pickle.dump(payload, fh)


@pytest.mark.parametrize("use_torch_save", [True, False], ids=["torch.save", "pickle.dump"])
def test_load_handles_both_serialisation_formats(tmp_path: Path, use_torch_save: bool) -> None:
    """EXP-4A is a bare pickle, which torch.load cannot read at all."""
    path = tmp_path / "ckpt.pkl"
    _write_efficientnet_checkpoint(path, units=1, use_torch_save=use_torch_save)

    settings = get_settings().model_copy(update={"model_path": path})
    model, info = load_model(settings)

    assert info.num_output_units == 1
    assert info.activation == "sigmoid"
    assert info.missing_keys == [] and info.unexpected_keys == []
    assert info.class_names == {0: "Non-Cancer", 1: "Cancer"}
    expected_loader = "torch.load" if use_torch_save else "pickle.load"
    assert expected_loader in info.load_path
    assert model.training is False, "model must be in eval mode"


def test_load_strips_a_dataparallel_prefix(tmp_path: Path) -> None:
    path = tmp_path / "dp.pkl"
    _write_efficientnet_checkpoint(path, units=1, use_torch_save=False, prefix="module.")
    settings = get_settings().model_copy(update={"model_path": path})
    _, info = load_model(settings)
    assert "stripped 'module.'" in info.load_path
    assert info.missing_keys == [] and info.unexpected_keys == []


def test_load_detects_a_two_output_head(tmp_path: Path) -> None:
    path = tmp_path / "two.pkl"
    _write_efficientnet_checkpoint(path, units=2, use_torch_save=False)
    settings = get_settings().model_copy(update={"model_path": path})
    _, info = load_model(settings)
    assert info.num_output_units == 2
    assert info.activation == "softmax"


def test_warmup_runs_at_load(tmp_path: Path) -> None:
    path = tmp_path / "w.pkl"
    _write_efficientnet_checkpoint(path, units=1, use_torch_save=False)
    settings = get_settings().model_copy(update={"model_path": path})
    _, info = load_model(settings)
    assert info.warmup_ms > 0.0


def test_missing_file_gives_an_actionable_error() -> None:
    settings = get_settings().model_copy(update={"model_path": Path("/nope/absent.pkl")})
    with pytest.raises(ModelLoadError, match="MODEL_PATH"):
        load_model(settings)


def test_garbage_file_gives_an_actionable_error(tmp_path: Path) -> None:
    path = tmp_path / "junk.pkl"
    path.write_bytes(b"not a checkpoint at all")
    settings = get_settings().model_copy(update={"model_path": path})
    with pytest.raises(ModelLoadError, match="Could not read checkpoint"):
        load_model(settings)


def test_tolerant_unpickler_substitutes_missing_classes() -> None:
    """The stub path must survive a class that cannot be imported."""
    import io as _io

    from app.model_loader import _tolerant_pickle_module

    def global_ref(module: str, name: str) -> bytes:
        """Assemble a protocol-4 pickle that resolves `module.name`.

        Built by hand because `pickle.dumps` refuses to emit a reference it
        cannot itself look up -- which is exactly the situation under test.
        """
        def short_str(text: str) -> bytes:
            raw = text.encode()
            return b"\x8c" + bytes([len(raw)]) + raw + b"\x94"  # SHORT_BINUNICODE + MEMOIZE

        return (
            b"\x80\x04"                       # PROTO 4
            + short_str(module)
            + short_str(name)
            + b"\x93\x94"                     # STACK_GLOBAL + MEMOIZE
            + b"."                             # STOP
        )

    payload = global_ref("no_such_module_here", "MissingModelClass")

    # Baseline: the standard unpickler must fail on this.
    with pytest.raises((ModuleNotFoundError, ImportError)):
        pickle.loads(payload)

    # The tolerant one substitutes a stub instead.
    shim = _tolerant_pickle_module()
    recovered = shim.Unpickler(_io.BytesIO(payload)).load()
    assert recovered is not None, "unpickler should return a stub, not raise"
    assert recovered.__name__ == "Stub_MissingModelClass"


# --- the real checkpoint ---------------------------------------------------
REAL_CHECKPOINT = get_settings().model_path


@pytest.mark.skipif(
    not REAL_CHECKPOINT.is_file(), reason="EXP-4A checkpoint not present"
)
def test_real_exp4a_checkpoint_loads_cleanly() -> None:
    """Pins every finding from the Phase 0 inspection of the real model."""
    model, info = load_model()

    assert info.num_output_units == 1
    assert info.activation == "sigmoid"
    assert info.outputs_probability is False
    assert info.missing_keys == []
    assert info.unexpected_keys == []
    assert info.class_names == {0: "Non-Cancer", 1: "Cancer"}
    assert info.positive_class_index == 1
    assert info.positive_class_name == "Cancer"
    assert info.negative_class_name == "Non-Cancer"
    assert info.input_size == 224
    assert info.normalize_mean == (0.485, 0.456, 0.406)
    assert "pickle.load" in info.load_path
    assert model.training is False

    # And it must produce a real logit, not a probability.
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 1)


# --- checkpoint delivery (deployment) --------------------------------------
def test_existing_checkpoint_is_not_redownloaded(tmp_path: Path) -> None:
    from app.model_loader import ensure_checkpoint

    path = tmp_path / "present.pkl"
    path.write_bytes(b"already here")
    settings = get_settings().model_copy(
        update={"model_path": path, "model_url": "http://should.not.be.used/x.pkl"}
    )
    assert ensure_checkpoint(settings) == path
    assert path.read_bytes() == b"already here"


def test_missing_checkpoint_without_a_url_explains_both_options(tmp_path: Path) -> None:
    from app.model_loader import ensure_checkpoint

    settings = get_settings().model_copy(
        update={"model_path": tmp_path / "absent.pkl", "model_url": ""}
    )
    with pytest.raises(ModelLoadError, match="MODEL_URL"):
        ensure_checkpoint(settings)


def test_checkpoint_is_downloaded_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The deploy path: weights are gitignored, so a fresh instance fetches them."""
    import io as _io

    from app import model_loader

    payload = b"pretend checkpoint bytes"

    class FakeResponse(_io.BytesIO):
        headers = {"Content-Type": "application/octet-stream"}

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    monkeypatch.setattr(
        model_loader.urllib.request, "urlopen",
        lambda *a, **k: FakeResponse(payload),
    )

    target = tmp_path / "fetched.pkl"
    settings = get_settings().model_copy(
        update={"model_path": target, "model_url": "https://example.com/model.pkl"}
    )
    assert model_loader.ensure_checkpoint(settings) == target
    assert target.read_bytes() == payload
    assert not (tmp_path / "fetched.pkl.part").exists(), "temp file must be cleaned up"


def test_download_rejects_an_html_sharing_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Google Drive / Dropbox preview URL returns HTML, not the file."""
    import io as _io

    from app import model_loader

    class HtmlResponse(_io.BytesIO):
        headers = {"Content-Type": "text/html; charset=utf-8"}

        def __enter__(self) -> "HtmlResponse":
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    monkeypatch.setattr(
        model_loader.urllib.request, "urlopen",
        lambda *a, **k: HtmlResponse(b"<html>Sign in to download</html>"),
    )

    target = tmp_path / "html.pkl"
    settings = get_settings().model_copy(
        update={"model_path": target, "model_url": "https://drive.google.com/file/d/abc/view"}
    )
    with pytest.raises(ModelLoadError, match="HTML"):
        model_loader.ensure_checkpoint(settings)
    assert not target.exists()


def test_download_verifies_the_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib as _hashlib
    import io as _io

    from app import model_loader

    payload = b"checkpoint contents"
    correct = _hashlib.sha256(payload).hexdigest()

    class FakeResponse(_io.BytesIO):
        headers = {"Content-Type": "application/octet-stream"}

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    monkeypatch.setattr(
        model_loader.urllib.request, "urlopen",
        lambda *a, **k: FakeResponse(payload),
    )

    base = {"model_url": "https://example.com/m.pkl"}

    good = get_settings().model_copy(
        update={**base, "model_path": tmp_path / "good.pkl", "model_sha256": correct}
    )
    assert model_loader.ensure_checkpoint(good).read_bytes() == payload

    bad = get_settings().model_copy(
        update={**base, "model_path": tmp_path / "bad.pkl", "model_sha256": "0" * 64}
    )
    with pytest.raises(ModelLoadError, match="integrity check"):
        model_loader.ensure_checkpoint(bad)
    assert not (tmp_path / "bad.pkl").exists(), "a corrupt download must not be kept"


# --- CORS configuration ----------------------------------------------------
def test_cors_regex_is_omitted_when_unset() -> None:
    """An empty regex would compile to a pattern matching every origin."""
    settings = get_settings().model_copy(update={"allowed_origin_regex": ""})
    assert "allow_origin_regex" not in settings.cors_kwargs


def test_cors_regex_is_included_when_set() -> None:
    pattern = r"https://myapp-.*\.vercel\.app"
    settings = get_settings().model_copy(update={"allowed_origin_regex": pattern})
    assert settings.cors_kwargs["allow_origin_regex"] == pattern
