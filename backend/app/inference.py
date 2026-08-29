"""Forward pass and adaptive post-processing.

The activation applied depends on what the checkpoint actually produces --
determined at load time, not assumed here:

    num_output_units == 1  ->  sigmoid(logit)          = P(positive)
    num_output_units == 2  ->  softmax(logits)[pos_ix] = P(positive)
    outputs_probability    ->  no activation; the value is already P

EXP-4A takes the first branch: a single raw logit + sigmoid.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
from fastapi.concurrency import run_in_threadpool
from torch import nn

from .exceptions import InferenceError
from .model_loader import ModelInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """One image's verdict, with everything needed to audit it."""

    prediction: str
    probability: float       # P(positive class) == P(Cancer)
    confidence: float        # probability of whichever class was predicted
    threshold: float
    raw_output: float        # the pre-activation logit (or raw value)
    is_positive: bool
    inference_time_ms: float


def _to_probability(output: torch.Tensor, info: ModelInfo) -> tuple[float, float]:
    """Reduce a model output to (P(positive), representative raw value)."""
    # Collapse a leading batch dimension of 1; keep the class dimension.
    flat = output.detach().float().reshape(-1)

    if flat.numel() == 0:
        raise InferenceError("The model returned an empty tensor.")

    if info.outputs_probability:
        # The model already applied its own activation; do not apply another.
        if flat.numel() == 1:
            probability = float(flat[0])
        else:
            probability = float(flat[info.positive_class_index])
        return probability, probability

    if flat.numel() == 1:
        # Single logit: sigmoid gives P(positive) directly.
        logit = float(flat[0])
        return float(torch.sigmoid(flat)[0]), logit

    if flat.numel() == 2:
        # Two logits: softmax over the class dimension, then take the positive.
        probs = torch.softmax(flat, dim=0)
        return float(probs[info.positive_class_index]), float(flat[info.positive_class_index])

    raise InferenceError(
        f"The model returned {flat.numel()} values, which is not a binary "
        "output. This app serves binary Cancer / Non-Cancer classification only."
    )


def _forward(model: nn.Module, batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Blocking forward pass. Always called off the event loop."""
    with torch.inference_mode():
        return model(batch.to(device))


def run_inference_sync(
    model: nn.Module, info: ModelInfo, batch: torch.Tensor
) -> PredictionResult:
    """Synchronous predict. Used directly by tests; wrapped for the API."""
    device = torch.device(info.device)
    start = time.perf_counter()
    try:
        output = _forward(model, batch, device)
    except InferenceError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(
            f"The model failed to process the image: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(output, torch.Tensor):
        raise InferenceError(
            f"The model returned {type(output).__name__}, not a tensor."
        )

    probability, raw = _to_probability(output, info)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if not 0.0 <= probability <= 1.0:
        # Only reachable when MODEL_OUTPUTS_PROBABILITY is wrongly set to true.
        raise InferenceError(
            f"Computed probability {probability:.4f} is outside [0, 1]. "
            "MODEL_OUTPUTS_PROBABILITY is probably set incorrectly -- the model "
            "appears to emit raw logits that still need an activation."
        )

    is_positive = probability >= info.threshold
    label = info.positive_class_name if is_positive else info.negative_class_name

    return PredictionResult(
        prediction=label,
        probability=probability,
        # Confidence is in the PREDICTED class, so it is never below 0.5 at a
        # 0.5 threshold -- it answers "how sure about this label", not "P(cancer)".
        confidence=probability if is_positive else 1.0 - probability,
        threshold=info.threshold,
        raw_output=raw,
        is_positive=is_positive,
        inference_time_ms=round(elapsed_ms, 2),
    )


async def run_inference(
    model: nn.Module, info: ModelInfo, batch: torch.Tensor
) -> PredictionResult:
    """Async wrapper: the forward pass runs in a worker thread.

    PyTorch releases the GIL during the heavy ops, so this keeps the event loop
    responsive while a request is being scored.
    """
    return await run_in_threadpool(run_inference_sync, model, info, batch)
