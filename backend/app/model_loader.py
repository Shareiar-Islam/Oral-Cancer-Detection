"""Load the checkpoint once, defensively, and expose it as a singleton.

The checkpoint is unpickled at startup inside the FastAPI lifespan and held in
module state. Nothing here runs per request.

Cases handled, in the order they are tried:

  1. A full ``nn.Module`` was pickled  -> use it directly.
  2. A ``state_dict`` (flat, or nested under ``state_dict`` /
     ``model_state_dict`` / ``model`` / ...) -> rebuild torchvision's
     efficientnet_b0, resize the head to match, load with ``strict=False``.
  3. The pickle references a class that no longer exists
     (``ModuleNotFoundError`` / ``AttributeError``) -> re-unpickle with a
     tolerant Unpickler that substitutes stubs for the missing classes, then
     recover the tensors from whatever comes back and fall back to case 2.

EXP-4A takes path 2: a state_dict nested under "state_dict", no key prefix,
``classifier.1.weight`` of shape (1, 1280) -> single logit + sigmoid.
"""

from __future__ import annotations

import hashlib
import io
import logging
import pickle
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models

from .config import Settings, get_settings
from .exceptions import ModelLoadError, ModelNotLoadedError

logger = logging.getLogger(__name__)

# Wrapper prefixes that must be stripped before torchvision will accept keys.
KNOWN_PREFIXES: tuple[str, ...] = ("module.", "model.", "_orig_mod.")

# Keys a state_dict is commonly nested under inside a checkpoint bundle.
NESTED_STATE_DICT_KEYS: tuple[str, ...] = (
    "state_dict",
    "model_state_dict",
    "model",
    "net",
    "weights",
)

# Metadata keys EXP-4A records; read in preference to .env so that serving
# config can never silently disagree with how the model was trained.
META_CLASS_NAMES = ("class_names", "classes", "class_to_idx", "idx_to_class")

EFFICIENTNET_B0_FEATURES = 1280


@dataclass
class ModelInfo:
    """Everything the app discovered about the loaded model."""

    architecture: str = "torchvision.models.efficientnet_b0"
    checkpoint_path: str = ""
    device: str = "cpu"
    num_output_units: int = 1
    activation: str = "sigmoid"
    outputs_probability: bool = False
    threshold: float = 0.5
    positive_class_index: int = 1
    class_names: dict[int, str] = field(default_factory=lambda: {0: "Non-Cancer", 1: "Cancer"})
    input_size: int = 224
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    resize_mode: str = "resize"
    load_path: str = "state_dict"
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    checkpoint_metadata: dict[str, Any] = field(default_factory=dict)
    warmup_ms: float = 0.0

    @property
    def positive_class_name(self) -> str:
        return self.class_names.get(self.positive_class_index, "Cancer")

    @property
    def negative_class_name(self) -> str:
        other = 1 - self.positive_class_index
        return self.class_names.get(other, "Non-Cancer")


# --------------------------------------------------------------------------
# Tolerant unpickling (case 3)
# --------------------------------------------------------------------------
class _MissingClassStub:
    """Stand-in for a class the pickle references but we cannot import.

    Unpickling only needs *something* to hang the instance ``__dict__`` on.
    The tensors we actually care about are recovered from that dict afterwards.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args = args
        self._kwargs = kwargs

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state

    # Pickle may call these on the stub while rebuilding an nn.Module.
    def append(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        return None


def _tolerant_pickle_module() -> Any:
    """A ``pickle``-compatible module whose Unpickler never fails on imports."""

    class _TolerantUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError, ImportError):
                logger.warning(
                    "Substituting a stub for unresolvable pickle class %s.%s",
                    module,
                    name,
                )
                return type(f"Stub_{name}", (_MissingClassStub,), {})

    class _Shim:
        Unpickler = _TolerantUnpickler
        UnpicklingError = pickle.UnpicklingError

        @staticmethod
        def load(file: Any, **kwargs: Any) -> Any:
            return _TolerantUnpickler(file).load()

        @staticmethod
        def loads(data: bytes, **kwargs: Any) -> Any:
            return _TolerantUnpickler(io.BytesIO(data)).load()

    return _Shim


def _walk_for_state_dict(obj: Any, depth: int = 0) -> dict[str, torch.Tensor] | None:
    """Recursively hunt for the largest tensor mapping inside an object graph."""
    if depth > 6:
        return None

    if isinstance(obj, dict):
        tensors = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
        # A real state_dict is overwhelmingly tensors; a config dict is not.
        # Require at least 2 so a lone tensor sitting in a metadata dict (a
        # recorded threshold, say) is not mistaken for a set of weights.
        if len(tensors) >= 2 and len(tensors) >= len(obj) // 2:
            return tensors
        best: dict[str, torch.Tensor] | None = None
        for value in obj.values():
            found = _walk_for_state_dict(value, depth + 1)
            if found and (best is None or len(found) > len(best)):
                best = found
        return best

    if hasattr(obj, "__dict__"):
        return _walk_for_state_dict(vars(obj), depth + 1)

    return None


# --------------------------------------------------------------------------
# Checkpoint dissection
# --------------------------------------------------------------------------
def _unwrap_state_dict(obj: Any) -> tuple[dict[str, torch.Tensor] | None, str]:
    """Locate the tensor mapping and report where it was found."""
    if isinstance(obj, nn.Module):
        return obj.state_dict(), "nn.Module"

    if isinstance(obj, dict):
        values = list(obj.values())
        if values and all(isinstance(v, torch.Tensor) for v in values):
            return obj, "top-level"
        for key in NESTED_STATE_DICT_KEYS:
            inner = obj.get(key)
            if isinstance(inner, nn.Module):
                return inner.state_dict(), f"{key} (nn.Module)"
            if isinstance(inner, dict) and inner and all(
                isinstance(v, torch.Tensor) for v in inner.values()
            ):
                return inner, key

    found = _walk_for_state_dict(obj)
    return (found, "recovered by search") if found else (None, "not found")


def _strip_prefix(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], str | None]:
    """Remove a uniform ``module.`` / ``model.`` wrapper prefix, if present."""
    keys = list(state.keys())
    for prefix in KNOWN_PREFIXES:
        if keys and all(k.startswith(prefix) for k in keys):
            logger.info("Stripping '%s' prefix from %d state_dict keys.", prefix, len(keys))
            return {k[len(prefix):]: v for k, v in state.items()}, prefix
    return state, None


def _infer_num_outputs(state: dict[str, torch.Tensor]) -> int:
    """Number of units in the final classifier -- 1 => sigmoid, 2 => softmax."""
    head_hints = ("classifier", "fc.", "head", "last_linear", "_fc")
    candidates = [
        (k, tuple(v.shape))
        for k, v in state.items()
        if k.endswith("weight") and isinstance(v, torch.Tensor) and v.ndim == 2
    ]
    if not candidates:
        raise ModelLoadError(
            "No 2-D classifier weight found in the checkpoint, so the number "
            "of output units cannot be determined. Run "
            "`python scripts/inspect_model.py <path>` to inspect it."
        )
    for key, shape in reversed(candidates):
        if any(hint in key.lower() for hint in head_hints):
            logger.info("Classifier head: %s shape=%s", key, shape)
            return int(shape[0])
    key, shape = candidates[-1]
    logger.info("Classifier head (fallback, last 2-D weight): %s shape=%s", key, shape)
    return int(shape[0])


def _build_backbone(num_outputs: int) -> nn.Module:
    """EfficientNet-B0 with the head resized to ``num_outputs``.

    ASSUMPTION (matches EXP-4A, which loads with 0 missing / 0 unexpected keys):
    training replaced ``classifier[1]`` with ``nn.Linear(1280, num_outputs)``,
    leaving the dropout at ``classifier[0]`` in place.
    """
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    if in_features != EFFICIENTNET_B0_FEATURES:  # pragma: no cover - torchvision invariant
        logger.warning("Unexpected feature width %d (expected %d).", in_features, EFFICIENTNET_B0_FEATURES)
    model.classifier[1] = nn.Linear(in_features, num_outputs)
    return model


def _extract_metadata(obj: Any) -> dict[str, Any]:
    """Pull the non-tensor bookkeeping the training script recorded."""
    if not isinstance(obj, dict):
        return {}
    meta: dict[str, Any] = {}
    for key, value in obj.items():
        if isinstance(value, torch.Tensor) or isinstance(value, nn.Module):
            continue
        if isinstance(value, dict) and value and all(
            isinstance(v, torch.Tensor) for v in value.values()
        ):
            continue  # that's the weights, not metadata
        meta[key] = value
    return meta


def _parse_class_names(meta: dict[str, Any]) -> dict[int, str] | None:
    """Normalise whatever label mapping the checkpoint recorded to {index: name}.

    Handles ``{0: 'NON CANCER', 1: 'CANCER'}``, the inverted
    ``{'non_cancer': 0, ...}`` (torchvision's ``class_to_idx``), and a plain
    ordered list of names.
    """
    for key in META_CLASS_NAMES:
        raw = meta.get(key)
        if raw is None:
            continue
        if isinstance(raw, dict):
            items = list(raw.items())
            if all(isinstance(k, int) for k, _ in items):
                return {int(k): str(v).strip() for k, v in items}
            if all(isinstance(v, int) for _, v in items):  # class_to_idx
                return {int(v): str(k).strip() for k, v in items}
            # String keys that are really digits, e.g. JSON round-tripped.
            try:
                return {int(k): str(v).strip() for k, v in items}
            except (TypeError, ValueError):
                continue
        if isinstance(raw, (list, tuple)) and all(isinstance(v, str) for v in raw):
            return {i: v.strip() for i, v in enumerate(raw)}
    return None


def _titlecase(name: str) -> str:
    """'NON CANCER' -> 'Non-Cancer'; 'CANCER' -> 'Cancer'."""
    cleaned = name.replace("_", " ").replace("-", " ").strip()
    parts = [p.capitalize() for p in cleaned.split() if p]
    if len(parts) == 2 and parts[0].lower() == "non":
        return f"Non-{parts[1]}"
    return " ".join(parts) or name


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def _redact(url: str) -> str:
    """Strip the query string so a presigned URL's signature stays out of logs."""
    return url.split("?", 1)[0] + ("?<redacted>" if "?" in url else "")


def ensure_checkpoint(settings: Settings) -> Path:
    """Make sure MODEL_PATH exists, downloading from MODEL_URL if it does not.

    Hosting platforms build from the git repo, and the weights are gitignored,
    so on a fresh deploy the file is simply absent. Rather than requiring the
    binary in git, fetch it once at boot and cache it on the instance disk.

    Downloads to a temporary file and renames only on success, so an
    interrupted transfer can never leave a truncated checkpoint behind.
    """
    path = settings.model_path
    if path.is_file():
        return path

    url = settings.model_url.strip()
    if not url:
        raise ModelLoadError(
            f"No checkpoint at {path}, and MODEL_URL is not set. Either point "
            "MODEL_PATH at your .pkl file, or set MODEL_URL to a "
            "direct-download link so it can be fetched at startup."
        )

    logger.info("Checkpoint missing; downloading from %s", _redact(url))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".part")

    request = urllib.request.Request(url)
    if settings.model_auth_token.strip():
        # Private object stores, and Hugging Face private repos, need this.
        # A presigned URL carries its own credentials and needs no header.
        request.add_header("Authorization", f"Bearer {settings.model_auth_token.strip()}")
        logger.info("Using MODEL_AUTH_TOKEN for the checkpoint download.")

    try:
        with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type:
                raise ModelLoadError(
                    f"MODEL_URL returned HTML, not a file (Content-Type: "
                    f"{content_type}). Sharing pages from Google Drive or "
                    "Dropbox need a direct-download link instead."
                )
            digest = hashlib.sha256()
            with temp_path.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
    except ModelLoadError:
        temp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001
        temp_path.unlink(missing_ok=True)
        raise ModelLoadError(
            f"Failed to download the checkpoint from {_redact(url)}: "
            f"{type(exc).__name__}: {exc}. "
            "If the URL is private, set MODEL_AUTH_TOKEN."
        ) from exc

    expected = settings.model_sha256.strip().lower()
    if expected and digest.hexdigest() != expected:
        temp_path.unlink(missing_ok=True)
        raise ModelLoadError(
            f"Downloaded checkpoint failed its integrity check. Expected "
            f"sha256 {expected}, got {digest.hexdigest()}."
        )

    temp_path.rename(path)
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Downloaded checkpoint to %s (%.1f MB)", path, size_mb)
    return path


def _load_raw(path: Path, device: torch.device) -> tuple[Any, str]:
    """Unpickle the checkpoint, trying each serialisation format in turn.

    Returns (object, how_it_was_loaded).

    Two formats are in play and the extension does not tell them apart:
      * ``torch.save``  -> a zip archive; only ``torch.load`` reads it.
      * ``pickle.dump`` -> a bare pickle stream; ``torch.load`` rejects it with
        "Invalid magic number". EXP-4A is this one.

    ``weights_only=False`` / bare pickle both execute code from the file, so
    only ever point MODEL_PATH at a checkpoint you produced yourself.
    """
    if not path.is_file():
        raise ModelLoadError(
            f"No checkpoint at {path}. Set MODEL_PATH in backend/.env to point "
            "at your .pkl file. (load_model() calls ensure_checkpoint() first, "
            "so this is only reachable when _load_raw is called directly.)"
        )

    attempts: list[str] = []

    # 1. torch.save format.
    try:
        return torch.load(path, map_location=device, weights_only=False), "torch.load"
    except (ModuleNotFoundError, AttributeError, ImportError) as exc:
        attempts.append(f"torch.load: {type(exc).__name__}: {exc}")
        missing_class_error: Exception | None = exc
    except Exception as exc:  # noqa: BLE001 - wrong container format, try the next
        attempts.append(f"torch.load: {type(exc).__name__}: {exc}")
        missing_class_error = None

    # 2. Bare pickle stream (what a plain pickle.dump produces).
    try:
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        logger.info("Checkpoint read as a plain pickle stream (not torch.save format).")
        return obj, "pickle.load"
    except (ModuleNotFoundError, AttributeError, ImportError) as exc:
        attempts.append(f"pickle.load: {type(exc).__name__}: {exc}")
        missing_class_error = missing_class_error or exc
    except Exception as exc:  # noqa: BLE001
        attempts.append(f"pickle.load: {type(exc).__name__}: {exc}")

    # 3. Both failed on an unimportable class -> stub it out and recover tensors.
    if missing_class_error is not None:
        logger.warning(
            "Checkpoint references a class that is not importable here (%s). "
            "Retrying with a tolerant unpickler to recover the weights.",
            missing_class_error,
        )
        shim = _tolerant_pickle_module()
        for label, loader in (
            ("torch.load (tolerant)", lambda: torch.load(
                path, map_location=device, weights_only=False, pickle_module=shim)),
            ("pickle.load (tolerant)", lambda: shim.load(path.open("rb"))),
        ):
            try:
                return loader(), label
            except Exception as exc:  # noqa: BLE001
                attempts.append(f"{label}: {type(exc).__name__}: {exc}")

    detail = "\n  - ".join(attempts)
    raise ModelLoadError(
        f"Could not read checkpoint {path.name}. Attempts:\n  - {detail}\n"
        "If a class could not be imported, re-export from your training "
        "environment with: torch.save(model.state_dict(), 'weights.pth')"
    )


def load_model(settings: Settings | None = None) -> tuple[nn.Module, ModelInfo]:
    """Build a ready-to-serve model and describe what was loaded."""
    settings = settings or get_settings()
    path = ensure_checkpoint(settings)
    device = settings.resolved_device

    logger.info("Loading checkpoint %s onto %s", path, device)
    obj, load_method = _load_raw(path, device)

    meta = _extract_metadata(obj)
    state, location = _unwrap_state_dict(obj)

    info = ModelInfo(
        checkpoint_path=str(path),
        device=str(device),
        threshold=settings.threshold,
        positive_class_index=settings.positive_class_index,
        input_size=settings.input_size,
        normalize_mean=settings.normalize_mean,
        normalize_std=settings.normalize_std,
        resize_mode=settings.resize_mode,
        outputs_probability=settings.model_outputs_probability,
        checkpoint_metadata={
            k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool, list, dict))
        },
    )

    # --- materialise the module ------------------------------------------
    if isinstance(obj, nn.Module):
        model = obj
        info.load_path = f"pickled nn.Module via {load_method}"
        info.num_output_units = _infer_num_outputs(obj.state_dict())
    else:
        if state is None:
            raise ModelLoadError(
                f"{path.name} contains no recognisable weights. Run "
                "`python scripts/inspect_model.py <path>` to see what is inside."
            )
        state, prefix = _strip_prefix(state)
        info.num_output_units = _infer_num_outputs(state)
        model = _build_backbone(info.num_output_units)
        result = model.load_state_dict(state, strict=False)
        info.missing_keys = list(result.missing_keys)
        info.unexpected_keys = list(result.unexpected_keys)
        info.load_path = (
            f"state_dict from '{location}' via {load_method}"
            + (f" (stripped '{prefix}')" if prefix else "")
        )

        if result.missing_keys or result.unexpected_keys:
            logger.warning(
                "state_dict mismatch: %d missing, %d unexpected. missing=%s unexpected=%s",
                len(result.missing_keys),
                len(result.unexpected_keys),
                result.missing_keys[:10],
                result.unexpected_keys[:10],
            )
            if len(result.missing_keys) > len(state) // 2:
                raise ModelLoadError(
                    "The checkpoint does not match EfficientNet-B0: "
                    f"{len(result.missing_keys)} of {len(state)} weights are missing. "
                    "The architecture assumption is wrong."
                )
        else:
            logger.info("Clean load: 0 missing, 0 unexpected keys.")

    # --- reconcile config against what the checkpoint declares ------------
    _apply_checkpoint_metadata(info, meta, settings)

    model.eval()
    model.to(device)
    info.warmup_ms = _warmup(model, device, settings.input_size)

    logger.info(
        "Model ready: %s | %d output unit(s) | activation=%s | threshold=%.2f | "
        "positive index %d = '%s' | warmup %.1f ms",
        info.load_path,
        info.num_output_units,
        info.activation,
        info.threshold,
        info.positive_class_index,
        info.positive_class_name,
        info.warmup_ms,
    )
    return model, info


def _apply_checkpoint_metadata(info: ModelInfo, meta: dict[str, Any], settings: Settings) -> None:
    """Let the checkpoint's own record win over .env, and shout about conflicts."""
    # Class names -- the mapping that decides whether predictions are inverted.
    parsed = _parse_class_names(meta)
    if parsed:
        info.class_names = {i: _titlecase(n) for i, n in parsed.items()}
        logger.info("Class mapping from checkpoint: %s", info.class_names)
    else:
        logger.warning(
            "The checkpoint records no class mapping. Falling back to "
            "{0: 'Non-Cancer', 1: 'Cancer'} with POSITIVE_CLASS_INDEX=%d. "
            "VERIFY against your training notebook -- an inverted mapping "
            "produces confidently wrong predictions with no error.",
            info.positive_class_index,
        )

    # Preprocessing geometry / normalisation.
    size = meta.get("image_size")
    if isinstance(size, int) and size != info.input_size:
        logger.warning(
            "Checkpoint image_size=%d disagrees with INPUT_SIZE=%d; using the checkpoint's.",
            size, info.input_size,
        )
        info.input_size = size
    for key, attr in (("mean", "normalize_mean"), ("std", "normalize_std")):
        value = meta.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            recorded = tuple(float(v) for v in value)
            if recorded != getattr(info, attr):
                logger.warning(
                    "Checkpoint %s=%s disagrees with config %s; using the checkpoint's.",
                    key, recorded, getattr(info, attr),
                )
            setattr(info, attr, recorded)

    # Activation. The checkpoint declares 'sigmoid' + 'single_logit' for EXP-4A,
    # meaning sigmoid must be APPLIED here (the loss applied it during training).
    declared_activation = str(meta.get("activation", "")).lower()
    if info.outputs_probability:
        info.activation = "none (checkpoint already emits probabilities)"
    elif info.num_output_units == 1:
        info.activation = "sigmoid"
    elif info.num_output_units == 2:
        info.activation = "softmax"
    else:
        raise ModelLoadError(
            f"The classifier has {info.num_output_units} output units, which is "
            "not a binary head. This app serves binary Cancer / Non-Cancer only."
        )
    if declared_activation and not info.outputs_probability and declared_activation not in info.activation:
        logger.warning(
            "Checkpoint declares activation='%s' but the head shape implies '%s'. "
            "Using '%s' -- verify against your training script.",
            declared_activation, info.activation, info.activation,
        )

    # Decision threshold: .env wins only if it was explicitly changed.
    recorded_threshold = meta.get("decision_threshold")
    if isinstance(recorded_threshold, (int, float)):
        if abs(float(recorded_threshold) - settings.threshold) > 1e-9:
            logger.warning(
                "Checkpoint decision_threshold=%.3f but THRESHOLD=%.3f is set in "
                "config; using the configured value.",
                float(recorded_threshold), settings.threshold,
            )

    if info.positive_class_index >= len(info.class_names) and info.class_names:
        raise ModelLoadError(
            f"POSITIVE_CLASS_INDEX={info.positive_class_index} is out of range "
            f"for class mapping {info.class_names}."
        )


def _warmup(model: nn.Module, device: torch.device, size: int) -> float:
    """One dummy pass so the first real request doesn't pay lazy-init cost."""
    dummy = torch.zeros(1, 3, size, size, device=device)
    start = time.perf_counter()
    with torch.inference_mode():
        model(dummy)
    if device.type == "cuda":  # pragma: no cover - CPU-only CI
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


# --------------------------------------------------------------------------
# Singleton access
# --------------------------------------------------------------------------
_model: nn.Module | None = None
_info: ModelInfo | None = None


def initialise(settings: Settings | None = None) -> ModelInfo:
    """Load into module state. Called once from the FastAPI lifespan."""
    global _model, _info
    _model, _info = load_model(settings)
    return _info


def set_model(model: nn.Module, info: ModelInfo) -> None:
    """Inject a model directly. Used by the tests to avoid touching disk."""
    global _model, _info
    _model, _info = model, info


def teardown() -> None:
    """Drop references at shutdown."""
    global _model, _info
    _model, _info = None, None


def is_loaded() -> bool:
    return _model is not None and _info is not None


def get_model() -> nn.Module:
    if _model is None:
        raise ModelNotLoadedError(
            "The model is not loaded. Check the server logs for the startup error."
        )
    return _model


def get_info() -> ModelInfo:
    if _info is None:
        raise ModelNotLoadedError(
            "The model is not loaded. Check the server logs for the startup error."
        )
    return _info
