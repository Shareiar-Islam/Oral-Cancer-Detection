"""Standalone diagnostic for an unknown oral-cancer checkpoint (.pkl / .pth).

Run this BEFORE writing or trusting any inference code. It answers the three
questions that determine the whole post-processing branch:

  1. What kind of object is inside the pickle?
  2. How many output units does the classifier have?  (1 -> sigmoid, 2 -> softmax)
  3. Does a dummy forward pass return raw logits or already-activated
     probabilities?  (value range outside [0, 1] => raw logits)

Usage:
    python scripts/inspect_model.py backend/models/oral_cancer.pkl
    python scripts/inspect_model.py path/to/model.pkl --json report.json

The script never mutates the checkpoint and never writes to it.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

# Input geometry assumed by the rest of the app. Kept here as the single source
# of truth for the diagnostic; the serving code reads it from config.
DUMMY_SHAPE: tuple[int, int, int, int] = (1, 3, 224, 224)

# Prefixes that commonly wrap state_dict keys and must be stripped before a
# torchvision model will accept them.
KNOWN_PREFIXES: tuple[str, ...] = ("module.", "model.", "_orig_mod.")

# Keys under which a state_dict is commonly nested inside a checkpoint dict.
NESTED_STATE_DICT_KEYS: tuple[str, ...] = (
    "state_dict",
    "model_state_dict",
    "model",
    "net",
    "weights",
)

SEP = "=" * 72


def _rule(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def _fmt_type(obj: Any) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
def load_checkpoint(path: Path) -> Any:
    """Unpickle the checkpoint, trying torch.load then plain pickle.

    weights_only=False is required because we do not yet know whether the file
    holds a bare tensor dict or a full pickled nn.Module. This executes
    arbitrary code from the pickle -- only ever point it at your own file.
    """
    errors: list[str] = []
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        print("Loader : torch.load  (torch.save zip-archive format)")
        return obj
    except Exception as exc:  # noqa: BLE001 - diagnostic must report, not raise
        errors.append(f"torch.load: {type(exc).__name__}: {exc}")

    try:
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        # Worth calling out: torch.load cannot read this file at all, so the
        # serving code must fall back to a bare pickle.load the same way.
        print("Loader : pickle.load  (plain pickle stream -- NOT torch.save format)")
        print(f"         torch.load rejected it: {errors[0].split(': ', 1)[1]}")
        return obj
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pickle.load: {type(exc).__name__}: {exc}")

    print("Could not unpickle the checkpoint. Attempts:")
    for err in errors:
        print(f"  - {err}")
    print(
        "\nIf you see ModuleNotFoundError/AttributeError above, the pickle "
        "references a class from the training script that is not importable "
        "here. Re-export as a state_dict from the training environment:\n"
        "    torch.save(model.state_dict(), 'weights.pth')"
    )
    raise SystemExit(2)


# --------------------------------------------------------------------------
# 2. Classify the object
# --------------------------------------------------------------------------
def describe_object(obj: Any) -> dict[str, Any]:
    _rule("1. OBJECT IDENTITY")
    print(f"type(obj)      : {_fmt_type(obj)}")
    mro = [f"{c.__module__}.{c.__qualname__}" for c in type(obj).__mro__]
    print(f"MRO            : {' -> '.join(mro)}")

    kind = "unknown"
    if isinstance(obj, nn.Module):
        # fastai Learner subclasses nothing torch-y, so check it separately.
        kind = "nn_module"
    elif type(obj).__name__ == "Learner" or hasattr(obj, "dls"):
        kind = "fastai_learner"
    elif hasattr(obj, "predict_proba"):
        kind = "sklearn_like"
    elif isinstance(obj, dict):
        kind = "state_dict_or_checkpoint"

    _rule("2. CASE DETECTION")
    labels = {
        "nn_module": "Full nn.Module (a pickled model object) -- usable directly.",
        "fastai_learner": "fastai Learner export -- needs fastai installed to run.",
        "sklearn_like": "sklearn-style wrapper exposing .predict_proba().",
        "state_dict_or_checkpoint": "dict / OrderedDict -- state_dict or checkpoint bundle.",
        "unknown": "Unrecognised object; inspect manually.",
    }
    print(f"CASE: {kind}\n  -> {labels[kind]}")
    if kind == "nn_module":
        n_params = sum(p.numel() for p in obj.parameters())
        print(f"  parameter count: {n_params:,}")
    return {"kind": kind, "type": _fmt_type(obj), "mro": mro}


# --------------------------------------------------------------------------
# 3. state_dict analysis
# --------------------------------------------------------------------------
def unwrap_state_dict(obj: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Return (state_dict, key_it_was_nested_under) if one can be found."""
    if isinstance(obj, nn.Module):
        # A live module still deserves the head-shape analysis below.
        return obj.state_dict(), "(live nn.Module)"
    if not isinstance(obj, dict):
        return None, None

    values = list(obj.values())
    if values and all(isinstance(v, torch.Tensor) for v in values):
        return obj, None  # already a flat state_dict

    for key in NESTED_STATE_DICT_KEYS:
        inner = obj.get(key)
        if isinstance(inner, dict) and inner and all(
            isinstance(v, torch.Tensor) for v in inner.values()
        ):
            return inner, key
        if isinstance(inner, nn.Module):
            return inner.state_dict(), key
    return None, None


def detect_prefix(keys: list[str]) -> str | None:
    for prefix in KNOWN_PREFIXES:
        if keys and all(k.startswith(prefix) for k in keys):
            return prefix
    return None


def find_classifier_weight(state: dict[str, Any]) -> tuple[str | None, tuple[int, ...] | None]:
    """Find the final classifier weight and return (key, shape).

    Strategy: prefer a key that looks like a head ('classifier', 'fc', 'head'),
    otherwise fall back to the last 2-D weight tensor in insertion order --
    which for a sequentially-built model is the output layer.
    """
    head_hints = ("classifier", "fc.", "head", "last_linear", "_fc")
    candidates = [
        (k, tuple(v.shape))
        for k, v in state.items()
        if isinstance(v, torch.Tensor) and k.endswith("weight") and v.ndim == 2
    ]
    if not candidates:
        return None, None
    for key, shape in reversed(candidates):
        if any(hint in key.lower() for hint in head_hints):
            return key, shape
    return candidates[-1]


def describe_state_dict(obj: Any) -> dict[str, Any]:
    state, nested_under = unwrap_state_dict(obj)
    _rule("3. STATE_DICT ANALYSIS")

    if state is None:
        if isinstance(obj, dict):
            print("Top-level dict keys (no tensor dict found directly):")
            for k in list(obj.keys())[:20]:
                print(f"  {k}: {_fmt_type(obj[k])}")
        else:
            print("Not a dict -- skipping state_dict analysis.")
        return {"found": False}

    keys = list(state.keys())
    print(f"nested under key : {nested_under or '(top level)'}")
    print(f"tensor count     : {len(keys)}")
    print(f"total parameters : {sum(v.numel() for v in state.values()):,}")

    print("\nFirst 5 keys:")
    for k in keys[:5]:
        print(f"  {k:<55} {tuple(state[k].shape)}")
    print("Last 5 keys:")
    for k in keys[-5:]:
        print(f"  {k:<55} {tuple(state[k].shape)}")

    prefix = detect_prefix(keys)
    print(f"\ncommon key prefix: {prefix or 'none'}")
    if prefix == "module.":
        print("  -> saved from nn.DataParallel/DDP; strip 'module.' before loading.")
    elif prefix:
        print(f"  -> wrapped module; strip '{prefix}' before loading.")

    head_key, head_shape = find_classifier_weight(state)
    num_outputs: int | None = None
    if head_shape is not None:
        num_outputs = head_shape[0]
        print(f"\nfinal classifier weight: {head_key}  shape={head_shape}")
        print(f"  in_features  : {head_shape[1]}   (EfficientNet-B0 expects 1280)")
        print(f"  OUTPUT UNITS : {num_outputs}")
        if num_outputs == 1:
            print("  => SINGLE LOGIT. Post-process with torch.sigmoid() = P(cancer).")
        elif num_outputs == 2:
            print("  => TWO LOGITS. Post-process with softmax(dim=1)[:, POSITIVE_CLASS_INDEX].")
        else:
            print(f"  => {num_outputs} units -- NOT a binary head. Check the checkpoint.")
    else:
        print("\nCould not identify a final classifier weight.")

    # Non-tensor metadata often carries the label order -- the single most
    # valuable thing in the file for catching an inverted class mapping.
    if isinstance(obj, dict) and not isinstance(obj, nn.Module):
        # Skip only the tensor blob itself -- class_to_idx / classes / label maps
        # are usually plain dicts or lists, so those must NOT be filtered out.
        def _is_tensor_blob(v: Any) -> bool:
            return isinstance(v, dict) and bool(v) and all(
                isinstance(t, torch.Tensor) for t in v.values()
            )

        extras = {
            k: v
            for k, v in obj.items()
            if not isinstance(v, torch.Tensor)
            and not _is_tensor_blob(v)
            and not isinstance(v, nn.Module)
        }
        if extras:
            _rule("3b. CHECKPOINT METADATA (look here for label order!)")
            for k, v in extras.items():
                text = repr(v)
                print(f"  {k}: {text[:300]}{'...' if len(text) > 300 else ''}")

    return {
        "found": True,
        "nested_under": nested_under,
        "num_tensors": len(keys),
        "prefix": prefix,
        "classifier_key": head_key,
        "classifier_shape": list(head_shape) if head_shape else None,
        "num_outputs": num_outputs,
    }


# --------------------------------------------------------------------------
# 4. Rebuild + dummy forward pass
# --------------------------------------------------------------------------
def build_efficientnet_b0(num_outputs: int) -> nn.Module:
    """EfficientNet-B0 with the classifier resized to num_outputs.

    ASSUMPTION (verify against your training notebook): the head was replaced
    as `model.classifier[1] = nn.Linear(1280, num_outputs)`, leaving the
    dropout at classifier[0] intact. If training instead appended extra dense
    layers, load_state_dict will report the mismatch as missing/unexpected keys.
    """
    from torchvision import models

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_outputs)
    return model


def materialise_model(obj: Any, analysis: dict[str, Any]) -> nn.Module | None:
    if isinstance(obj, nn.Module):
        return obj

    state, _ = unwrap_state_dict(obj)
    if state is None or not analysis.get("num_outputs"):
        return None

    prefix = analysis.get("prefix")
    if prefix:
        state = {k[len(prefix):]: v for k, v in state.items()}

    model = build_efficientnet_b0(int(analysis["num_outputs"]))
    result = model.load_state_dict(state, strict=False)
    _rule("4. RECONSTRUCTION (torchvision efficientnet_b0 + resized head)")
    print(f"missing keys    : {len(result.missing_keys)}")
    for k in result.missing_keys[:10]:
        print(f"    - {k}")
    print(f"unexpected keys : {len(result.unexpected_keys)}")
    for k in result.unexpected_keys[:10]:
        print(f"    + {k}")
    if not result.missing_keys and not result.unexpected_keys:
        print("  => CLEAN LOAD. Architecture matches the checkpoint exactly.")
    else:
        print(
            "  => Mismatch. If these are only head keys, the head shape differs; "
            "if they are backbone keys, the architecture is not efficientnet_b0."
        )
    return model


def find_trailing_activation(model: nn.Module) -> str | None:
    """Return the name of a Sigmoid/Softmax that sits at the end of the model.

    Structural evidence beats numeric guessing: if the module tree ends in an
    activation, the forward pass definitely returns probabilities. Absence is
    NOT proof of the opposite -- an activation applied functionally inside
    forward() (torch.sigmoid(x)) leaves no module behind.
    """
    leaves = [m for m in model.modules() if not list(m.children())]
    if leaves and isinstance(leaves[-1], (nn.Sigmoid, nn.Softmax, nn.LogSoftmax)):
        return type(leaves[-1]).__name__
    return None


def dummy_forward(model: nn.Module) -> dict[str, Any]:
    """Probe the model and decide whether it returns logits or probabilities.

    A single zeros input is NOT enough: an untrained or well-calibrated
    logit head can easily emit a value that happens to land inside [0, 1],
    which would falsely read as "already activated". So we push several
    inputs of increasing magnitude through the model -- a raw-logit head will
    escape [0, 1] on at least one of them, while a genuine sigmoid/softmax
    output cannot escape it on any.
    """
    _rule(f"5. DUMMY FORWARD PASS  input={DUMMY_SHAPE}")
    model.eval()

    torch.manual_seed(0)
    probes: list[tuple[str, torch.Tensor]] = [
        ("zeros", torch.zeros(*DUMMY_SHAPE)),
        ("randn", torch.randn(*DUMMY_SHAPE)),
        ("randn x5", torch.randn(*DUMMY_SHAPE) * 5.0),
        ("ones x10", torch.ones(*DUMMY_SHAPE) * 10.0),
        ("ones x-10", torch.ones(*DUMMY_SHAPE) * -10.0),
    ]

    outs: list[tuple[str, torch.Tensor]] = []
    with torch.no_grad():
        for name, x in probes:
            out = model(x)
            if not isinstance(out, torch.Tensor):
                print(f"Output is not a Tensor but {_fmt_type(out)}: {out!r}")
                return {"ok": False, "type": _fmt_type(out)}
            outs.append((name, out))

    first = outs[0][1]
    n_out = first.shape[-1] if first.ndim > 1 else 1
    print(f"output shape : {tuple(first.shape)}")
    print(f"output dtype : {first.dtype}")
    print(f"output units : {n_out}")

    print("\nprobe results:")
    global_lo, global_hi = float("inf"), float("-inf")
    all_sum_to_one = n_out == 2
    for name, out in outs:
        lo, hi = out.min().item(), out.max().item()
        global_lo, global_hi = min(global_lo, lo), max(global_hi, hi)
        total = out.sum().item()
        if n_out == 2 and abs(total - 1.0) > 1e-4:
            all_sum_to_one = False
        vals = ", ".join(f"{v:+.5f}" for v in out.flatten().tolist()[:4])
        extra = f"  sum={total:+.5f}" if n_out == 2 else ""
        print(f"  {name:<10} [{lo:+.5f}, {hi:+.5f}]  values=[{vals}]{extra}")

    print(f"\nrange across all probes : [{global_lo:.6f}, {global_hi:.6f}]")

    module_activation = find_trailing_activation(model)
    print(f"trailing activation module: {module_activation or 'none found'}")

    escaped_unit_range = global_lo < 0.0 or global_hi > 1.0
    # Pre-activated only if nothing ever left [0,1] AND we have corroborating
    # evidence: a trailing activation module, or (2-unit case) outputs that sum
    # to exactly 1 on every probe. Staying in [0,1] by luck is not enough.
    corroborated = bool(module_activation) or all_sum_to_one
    pre_activated = (not escaped_unit_range) and corroborated

    print("\nINTERPRETATION")
    if escaped_unit_range:
        print("  At least one probe left [0,1] -> output is RAW LOGITS. Certain.")
    elif pre_activated:
        why = f"trailing {module_activation}" if module_activation else "outputs sum to 1.0 on every probe"
        print(f"  Every probe stayed in [0,1] and {why} -> ALREADY ACTIVATED.")
    else:
        print("  Every probe stayed inside [0,1], but nothing corroborates an")
        print("  activation (no trailing Sigmoid/Softmax module"
              + (", outputs do not sum to 1" if n_out == 2 else "") + ").")
        print("  AMBIGUOUS -- treating as RAW LOGITS, which is the common case.")
        print("  Verify against your training script: did the final layer, the")
        print("  forward(), or only the loss function (BCEWithLogitsLoss /")
        print("  CrossEntropyLoss both expect LOGITS) apply the activation?")

    if pre_activated:
        print("\n  => set MODEL_OUTPUTS_PROBABILITY=true (skip sigmoid/softmax).")
    else:
        print("\n  => set MODEL_OUTPUTS_PROBABILITY=false.")
        sample = outs[1][1]  # the randn probe, more representative than zeros
        if n_out == 1:
            print(f"  => apply sigmoid; e.g. sigmoid({sample.flatten()[0].item():+.4f})"
                  f" = {torch.sigmoid(sample).flatten()[0].item():.6f}")
        elif n_out == 2:
            probs = torch.softmax(sample, dim=1).flatten().tolist()
            print(f"  => apply softmax(dim=1); e.g. probs = {[round(v, 6) for v in probs]}")

    print(
        "\nNOTE: these are synthetic inputs, not real images, so the predicted "
        "values are meaningless -- only SHAPE and RANGE matter here."
    )
    return {
        "ok": True,
        "shape": list(first.shape),
        "dtype": str(first.dtype),
        "min": global_lo,
        "max": global_hi,
        "num_outputs": int(n_out),
        "trailing_activation": module_activation,
        "looks_pre_activated": pre_activated,
        "ambiguous": (not escaped_unit_range) and not corroborated,
    }


# --------------------------------------------------------------------------
# 6. Recommendation
# --------------------------------------------------------------------------
def summarise(analysis: dict[str, Any], fwd: dict[str, Any] | None) -> dict[str, Any]:
    _rule("6. RECOMMENDED CONFIGURATION")
    n = (fwd or {}).get("num_outputs") or analysis.get("num_outputs")
    pre = bool((fwd or {}).get("looks_pre_activated"))

    if n == 1:
        branch = "sigmoid"
    elif n == 2:
        branch = "softmax"
    else:
        branch = "UNKNOWN"
    if pre:
        branch = "none (already activated)"
    if (fwd or {}).get("ambiguous"):
        branch += "   [AMBIGUOUS - verify against training script]"

    print(f"  output units          : {n}")
    print(f"  activation to apply   : {branch}")
    print(f"  MODEL_OUTPUTS_PROBABILITY = {'true' if pre else 'false'}")
    print("  POSITIVE_CLASS_INDEX      = 1   # TODO: verify against training label order")
    print("  THRESHOLD                 = 0.5")
    print(
        "\n  The class mapping is the ONE thing this script cannot determine. "
        "Confirm index 1 == Cancer by running a few known-label images through "
        "/api/predict (see the 'Verifying your model' section of the README)."
    )
    return {"num_outputs": n, "activation": branch, "outputs_probability": pre}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="path to the .pkl / .pth checkpoint")
    parser.add_argument("--json", type=Path, default=None, help="also write findings as JSON")
    args = parser.parse_args()

    if not args.path.is_file():
        print(f"No such file: {args.path}", file=sys.stderr)
        return 2

    size_mb = args.path.stat().st_size / (1024 * 1024)
    print(f"File   : {args.path.resolve()}")
    print(f"Size   : {size_mb:.2f} MB")
    print(f"torch  : {torch.__version__}")

    obj = load_checkpoint(args.path)
    identity = describe_object(obj)
    analysis = describe_state_dict(obj)

    model = materialise_model(obj, analysis)
    fwd: dict[str, Any] | None = None
    if model is None:
        _rule("4-5. RECONSTRUCTION / FORWARD PASS")
        print("Skipped: could not materialise an nn.Module from this object.")
        if identity["kind"] == "fastai_learner":
            print("  fastai Learner -> use learn.model to get the underlying nn.Module.")
        elif identity["kind"] == "sklearn_like":
            print("  sklearn-like -> call .predict_proba(X) instead of a torch forward pass.")
    else:
        try:
            fwd = dummy_forward(model)
        except Exception as exc:  # noqa: BLE001
            _rule("5. DUMMY FORWARD PASS")
            print(f"Forward pass failed: {type(exc).__name__}: {exc}")
            print("  -> the input shape or architecture assumption is wrong.")

    summary = summarise(analysis, fwd)

    if args.json:
        report = {"file": str(args.path), "identity": identity,
                  "state_dict": analysis, "forward": fwd, "summary": summary}
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nJSON report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
