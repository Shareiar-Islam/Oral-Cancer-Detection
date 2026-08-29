"""Post-processing: the sigmoid / softmax / pre-activated branches.

These are the tests that catch an inverted class mapping or a missing
activation -- the two failures that otherwise produce confident nonsense
with no error anywhere.
"""

from __future__ import annotations

import math

import pytest
import torch

from app.exceptions import InferenceError
from app.inference import run_inference_sync
from tests.conftest import ConstantModel, make_info


def run(output: torch.Tensor, **info_overrides: object):
    info = make_info(**info_overrides)
    model = ConstantModel(output)
    model.eval()
    return run_inference_sync(model, info, torch.zeros(1, 3, 224, 224))


# --- single logit -> sigmoid (the EXP-4A branch) ---------------------------
@pytest.mark.parametrize(
    "logit", [-8.0, -1.9231, -0.5, 0.0, 0.5, 1.9231, 8.0]
)
def test_single_logit_uses_sigmoid(logit: float) -> None:
    result = run(torch.tensor([[logit]]), num_output_units=1)
    assert result.probability == pytest.approx(1 / (1 + math.exp(-logit)), abs=1e-6)
    assert result.raw_output == pytest.approx(logit)


def test_single_logit_shape_1d_also_works() -> None:
    """Shape (1,) must behave identically to (1, 1)."""
    assert run(torch.tensor([1.9231])).probability == pytest.approx(
        run(torch.tensor([[1.9231]])).probability
    )


def test_positive_logit_predicts_cancer() -> None:
    result = run(torch.tensor([[1.9231]]))
    assert result.prediction == "Cancer"
    assert result.is_positive is True
    assert result.probability > 0.5


def test_negative_logit_predicts_non_cancer() -> None:
    result = run(torch.tensor([[-1.9231]]))
    assert result.prediction == "Non-Cancer"
    assert result.is_positive is False
    assert result.probability < 0.5


# --- two logits -> softmax -------------------------------------------------
def test_two_logits_use_softmax() -> None:
    logits = torch.tensor([[0.4, 2.1]])
    result = run(logits, num_output_units=2)
    expected = torch.softmax(logits, dim=1)[0, 1].item()
    assert result.probability == pytest.approx(expected, abs=1e-6)
    assert result.prediction == "Cancer"


def test_softmax_probabilities_are_complementary() -> None:
    """P from index 1 and P from index 0 must sum to 1 -- proves softmax, not sigmoid."""
    logits = torch.tensor([[1.2, -0.3]])
    p_one = run(logits, num_output_units=2, positive_class_index=1).probability
    p_zero = run(logits, num_output_units=2, positive_class_index=0).probability
    assert p_one + p_zero == pytest.approx(1.0, abs=1e-6)


def test_sigmoid_and_softmax_branches_differ() -> None:
    """Guard that the branch is chosen by output shape, not hardcoded.

    Note softmax([a, b])[1] == sigmoid(b - a), so a pair like [0, 2] would give
    the SAME answer either way and prove nothing. [1, 2] separates them:
    softmax -> sigmoid(1) = 0.731, whereas sigmoid(2) = 0.881.
    """
    two = run(torch.tensor([[1.0, 2.0]]), num_output_units=2).probability
    one = run(torch.tensor([[2.0]]), num_output_units=1).probability
    assert two == pytest.approx(1 / (1 + math.exp(-1.0)), abs=1e-6)
    assert one == pytest.approx(1 / (1 + math.exp(-2.0)), abs=1e-6)
    assert two != pytest.approx(one)


# --- pre-activated output --------------------------------------------------
def test_pre_activated_output_skips_sigmoid() -> None:
    """With MODEL_OUTPUTS_PROBABILITY=true the value is used as-is."""
    result = run(torch.tensor([[0.87]]), outputs_probability=True)
    assert result.probability == pytest.approx(0.87)
    assert result.prediction == "Cancer"


def test_pre_activated_two_output_takes_positive_index() -> None:
    result = run(
        torch.tensor([[0.3, 0.7]]), num_output_units=2, outputs_probability=True
    )
    assert result.probability == pytest.approx(0.7)


def test_wrongly_flagged_pre_activated_raises_not_silently_wrong() -> None:
    """A real logit outside [0,1] with outputs_probability=true must ERROR.

    This is the misconfiguration that would otherwise corrupt every result
    without a trace, so it has to be loud.
    """
    with pytest.raises(InferenceError, match="MODEL_OUTPUTS_PROBABILITY"):
        run(torch.tensor([[1.9231]]), outputs_probability=True)


# --- class mapping ---------------------------------------------------------
def test_positive_index_changes_what_probability_means_not_the_winner() -> None:
    """POSITIVE_CLASS_INDEX selects which class `probability` reports.

    It does NOT flip the verdict: index 1 clearly wins these logits, so both
    configurations still name class 1. What changes is that `probability`
    switches from P(class 1) to P(class 0), and `prediction` is then read off
    the opposite side of the threshold.
    """
    logits = torch.tensor([[0.2, 2.5]])
    normal = run(logits, num_output_units=2, positive_class_index=1)
    flipped = run(logits, num_output_units=2, positive_class_index=0)

    assert normal.prediction == "Cancer"
    assert flipped.prediction == "Cancer"          # same winner, either way
    assert normal.probability == pytest.approx(1 - flipped.probability, abs=1e-6)
    assert normal.probability > 0.5 > flipped.probability


def test_swapped_class_names_invert_the_verdict() -> None:
    """THIS is the silent bug the README's verification section guards against.

    Identical logits, identical maths -- only the recorded label order differs,
    and the reported diagnosis is the exact opposite. Nothing errors.
    """
    logits = torch.tensor([[3.0]])
    correct = run(logits, class_names={0: "Non-Cancer", 1: "Cancer"})
    inverted = run(logits, class_names={0: "Cancer", 1: "Non-Cancer"})

    assert correct.probability == pytest.approx(inverted.probability)
    assert correct.prediction == "Cancer"
    assert inverted.prediction == "Non-Cancer"


def test_custom_class_names_are_used() -> None:
    result = run(
        torch.tensor([[3.0]]), class_names={0: "Benign", 1: "Malignant"}
    )
    assert result.prediction == "Malignant"


# --- threshold -------------------------------------------------------------
@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(0.1, "Cancer"), (0.5, "Cancer"), (0.73, "Cancer"), (0.74, "Non-Cancer"), (0.9, "Non-Cancer")],
)
def test_threshold_moves_the_decision_boundary(threshold: float, expected: str) -> None:
    # sigmoid(1.0) == 0.73106, so the flip sits between 0.73 and 0.74.
    result = run(torch.tensor([[1.0]]), threshold=threshold)
    assert result.prediction == expected
    assert result.threshold == threshold


def test_probability_exactly_at_threshold_is_positive() -> None:
    """>= threshold, not > -- pinned so the boundary can't drift."""
    result = run(torch.tensor([[0.0]]), threshold=0.5)  # sigmoid(0) == 0.5
    assert result.probability == pytest.approx(0.5)
    assert result.prediction == "Cancer"


# --- confidence ------------------------------------------------------------
def test_confidence_is_probability_when_cancer() -> None:
    result = run(torch.tensor([[2.0]]))
    assert result.confidence == pytest.approx(result.probability)


def test_confidence_is_inverted_when_non_cancer() -> None:
    result = run(torch.tensor([[-2.0]]))
    assert result.confidence == pytest.approx(1 - result.probability)
    assert result.confidence > 0.5


@pytest.mark.parametrize("logit", [-5.0, -1.0, 0.0, 1.0, 5.0])
def test_confidence_never_below_half_at_default_threshold(logit: float) -> None:
    assert run(torch.tensor([[logit]])).confidence >= 0.5 - 1e-9


# --- malformed output ------------------------------------------------------
def test_three_way_output_rejected() -> None:
    with pytest.raises(InferenceError, match="not a binary output"):
        run(torch.tensor([[0.1, 0.2, 0.7]]), num_output_units=3)


def test_empty_output_rejected() -> None:
    with pytest.raises(InferenceError, match="empty tensor"):
        run(torch.zeros(1, 0))


def test_non_tensor_output_rejected() -> None:
    class BadModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, str]:
            return {"logits": "oops"}

    with pytest.raises(InferenceError, match="not a tensor"):
        run_inference_sync(BadModel(), make_info(), torch.zeros(1, 3, 224, 224))


def test_timing_is_recorded() -> None:
    result = run(torch.tensor([[1.0]]))
    assert result.inference_time_ms >= 0.0


def test_inference_does_not_build_a_graph() -> None:
    """inference_mode must be active, or memory grows under load."""

    class GradCheckModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(3 * 224 * 224, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.linear(x.flatten(1))
            assert not out.requires_grad, "forward pass is building a grad graph"
            return out

    model = GradCheckModel()
    model.eval()
    run_inference_sync(model, make_info(), torch.zeros(1, 3, 224, 224))
