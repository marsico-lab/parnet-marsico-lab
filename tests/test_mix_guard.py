"""Tests for the mix-coefficient compatibility guard.

The guard exists because two specific pairings produce wrong numbers rather than errors.
Both are reproduced here as the thing that must raise.
"""

import pytest
import torch

from parnet import layers
from parnet.layers import (
    AdditiveMix,
    IncompatibleMixError,
    LinearProjection,
    LinearProjectionHead,
    MixCoeffMLP,
    MixCoeffPenalty,
    NewAdditiveMix,
    NewMixCoeffMLP,
    SequenceLinearMix,
    check_mix_compatibility,
    mix_coeff_range,
)

N = 8


def test_declared_ranges():
    assert mix_coeff_range(SequenceLinearMix(N)) == "logit"
    assert mix_coeff_range(MixCoeffMLP(N)) == "unit"
    assert mix_coeff_range(NewMixCoeffMLP(N)) == "unit"


# -- the two real released combinations must keep working ---------------------------

def test_released_21m_combination_is_accepted():
    # AdditiveMix + SequenceLinearMix (logit) with the log-sum branch.
    AdditiveMix(N, head_layer=LinearProjectionHead, mix_coeff_layer=SequenceLinearMix,
                use_maximum_target_control_logprob=False)


def test_3m_and_7m_combination_is_accepted():
    # AdditiveMix + MixCoeffMLP (unit) with the max-logprob branch, which is how build_3m
    # constructs the 3M backbones.
    AdditiveMix(N, head_layer=LinearProjectionHead, mix_coeff_layer=MixCoeffMLP,
                penalty_layer=MixCoeffPenalty, use_maximum_target_control_logprob=True)
    # NewAdditiveMix + NewMixCoeffMLP (unit), the 7M head.
    NewAdditiveMix(N, head_layer=LinearProjection, mix_coeff_layer=NewMixCoeffMLP)


# -- the two failure modes ------------------------------------------------------------

def test_logit_into_convex_raises():
    # Measured consequence: NaN logits, because (1 - w) goes large and negative.
    with pytest.raises(IncompatibleMixError, match="NaN"):
        AdditiveMix(N, mix_coeff_layer=SequenceLinearMix,
                    use_maximum_target_control_logprob=True)


def test_unit_into_logsum_raises():
    # Measured consequence: effective target weight confined to (0.5, 0.731).
    with pytest.raises(IncompatibleMixError, match=r"0.5, 0.731"):
        AdditiveMix(N, mix_coeff_layer=MixCoeffMLP,
                    use_maximum_target_control_logprob=False)


def test_additivemix_defaults_are_the_incompatible_pair():
    # This is the latent hazard: the class's own defaults pair a sigmoid layer with the
    # log-sum branch. Anyone writing AdditiveMix(num_tasks) got the double sigmoid.
    with pytest.raises(IncompatibleMixError):
        AdditiveMix(N)


def test_newadditivemix_rejects_a_logit_layer():
    with pytest.raises(IncompatibleMixError, match="NaN"):
        NewAdditiveMix(N, mix_coeff_layer=SequenceLinearMix)


# -- escape hatch and unknown layers --------------------------------------------------

def test_strict_mix_false_warns_instead_of_raising():
    with pytest.warns(UserWarning, match="mix_coeff"):
        AdditiveMix(N, mix_coeff_layer=MixCoeffMLP,
                    use_maximum_target_control_logprob=False, strict_mix=False)


def test_unknown_layer_passes_unchecked():
    class MyMix(torch.nn.Module):
        def __init__(self, num_tasks):
            super().__init__()
            self.d = torch.nn.LazyLinear(num_tasks)

        def forward(self, x):
            return self.d(x.mean(-1))

    AdditiveMix(N, mix_coeff_layer=MyMix)  # declares nothing, so no opinion is imposed


def test_user_layer_can_opt_in_and_then_is_checked():
    class MyLogitMix(torch.nn.Module):
        mix_coeff_range = "logit"

        def __init__(self, num_tasks):
            super().__init__()
            self.d = torch.nn.LazyLinear(num_tasks)

        def forward(self, x):
            return self.d(x.mean(-1))

    with pytest.raises(IncompatibleMixError):
        AdditiveMix(N, mix_coeff_layer=MyLogitMix, use_maximum_target_control_logprob=True)


def test_check_helper_is_a_no_op_when_ranges_agree():
    check_mix_compatibility(MixCoeffMLP(N), "unit", "test rule")


def test_mix_coeff_keeps_the_batch_axis_at_batch_size_one():
    """SequenceLinearMix used a bare torch.squeeze, which drops the batch axis when B=1.

    The predicted tracks were unaffected because the subsequent unsqueeze broadcast
    correctly either way, so this surfaced only as `IndexError: too many indices` in
    caller code that did out['mix_coeff'][0, task]. Shapes must not depend on batch size.
    """
    head = layers.AdditiveMix(
        num_tasks=5,
        mix_coeff_layer=layers.SequenceLinearMix,
        use_maximum_target_control_logprob=False,
    )
    head(torch.randn(2, 8, 16))  # materialise lazy parameters
    head.eval()

    with torch.no_grad():
        single = head(torch.randn(1, 8, 16))
        batched = head(torch.randn(3, 8, 16))

    assert single['mix_coeff'].shape == (1, 5), single['mix_coeff'].shape
    assert batched['mix_coeff'].shape == (3, 5), batched['mix_coeff'].shape
    assert single['total'].shape == (1, 5, 16)


def test_batching_does_not_change_predictions():
    """A window predicted alone and inside a batch must give the same answer."""
    head = layers.AdditiveMix(
        num_tasks=3,
        mix_coeff_layer=layers.SequenceLinearMix,
        use_maximum_target_control_logprob=False,
    )
    head(torch.randn(2, 8, 16))
    head.eval()

    torch.manual_seed(1)
    x = torch.randn(4, 8, 16)
    with torch.no_grad():
        alone = head(x[:1])
        together = head(x)
    assert torch.allclose(alone['total'], together['total'][:1], atol=1e-6)
    assert torch.allclose(alone['mix_coeff'], together['mix_coeff'][:1], atol=1e-6)
