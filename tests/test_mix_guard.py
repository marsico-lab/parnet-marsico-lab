"""Tests for the mix-coefficient compatibility guard.

The guard exists because two specific pairings produce wrong numbers rather than errors.
Both are reproduced here as the thing that must raise.
"""

import functools

import pytest
import torch

from parnet import layers
from parnet.layers import (
    ConvexTargetControlMix,
    IncompatibleMixError,
    LinearProjection,
    LinearProjectionHead,
    MixCoeffPenalty,
    PooledLogitMixCoeff,
    PooledProbMixCoeff,
    PooledProbMixCoeffV2,
    TargetControlMix,
    check_mix_compatibility,
    mix_coeff_range,
)

N = 8


def test_declared_ranges():
    assert mix_coeff_range(PooledLogitMixCoeff(N)) == 'logit'
    assert mix_coeff_range(PooledProbMixCoeff(N)) == 'unit'
    assert mix_coeff_range(PooledProbMixCoeffV2(N)) == 'unit'


# -- the two real released combinations must keep working ---------------------------


def test_released_21m_combination_is_accepted():
    # TargetControlMix + PooledLogitMixCoeff (logit) with the log-sum branch.
    TargetControlMix(
        N,
        head_layer=LinearProjectionHead,
        mix_coeff_layer=PooledLogitMixCoeff,
        use_maximum_target_control_logprob=False,
    )


def test_3m_and_7m_combination_is_accepted():
    # TargetControlMix + PooledProbMixCoeff (unit) with the max-logprob branch, which is how build_3m
    # constructs the 3M backbones.
    TargetControlMix(
        N,
        head_layer=LinearProjectionHead,
        mix_coeff_layer=PooledProbMixCoeff,
        penalty_layer=MixCoeffPenalty,
        use_maximum_target_control_logprob=True,
    )
    # ConvexTargetControlMix + PooledProbMixCoeffV2 (unit), the 7M head.
    ConvexTargetControlMix(N, head_layer=LinearProjection, mix_coeff_layer=PooledProbMixCoeffV2)


# -- the two failure modes ------------------------------------------------------------


def test_logit_into_convex_raises():
    # Measured consequence: NaN logits, because (1 - w) goes large and negative.
    with pytest.raises(IncompatibleMixError, match='NaN'):
        TargetControlMix(N, mix_coeff_layer=PooledLogitMixCoeff, use_maximum_target_control_logprob=True)


def test_unit_into_logsum_raises():
    # Measured consequence: effective target weight confined to (0.5, 0.731).
    with pytest.raises(IncompatibleMixError, match=r'0.5, 0.731'):
        TargetControlMix(N, mix_coeff_layer=PooledProbMixCoeff, use_maximum_target_control_logprob=False)


def test_additivemix_defaults_are_the_incompatible_pair():
    # This is the latent hazard: the class's own defaults pair a sigmoid layer with the
    # log-sum branch. Anyone writing TargetControlMix(num_tasks) got the double sigmoid.
    with pytest.raises(IncompatibleMixError):
        TargetControlMix(N)


def test_newadditivemix_rejects_a_logit_layer():
    with pytest.raises(IncompatibleMixError, match='NaN'):
        ConvexTargetControlMix(N, mix_coeff_layer=PooledLogitMixCoeff)


# -- escape hatch and unknown layers --------------------------------------------------


def test_strict_mix_false_warns_instead_of_raising():
    with pytest.warns(UserWarning, match='mix_coeff'):
        TargetControlMix(
            N, mix_coeff_layer=PooledProbMixCoeff, use_maximum_target_control_logprob=False, strict_mix=False
        )


def test_unknown_layer_passes_unchecked():
    class MyMix(torch.nn.Module):
        def __init__(self, num_tasks):
            super().__init__()
            self.d = torch.nn.LazyLinear(num_tasks)

        def forward(self, x):
            return self.d(x.mean(-1))

    TargetControlMix(N, mix_coeff_layer=MyMix)  # declares nothing, so no opinion is imposed


def test_user_layer_can_opt_in_and_then_is_checked():
    class MyLogitMix(torch.nn.Module):
        mix_coeff_range = 'logit'

        def __init__(self, num_tasks):
            super().__init__()
            self.d = torch.nn.LazyLinear(num_tasks)

        def forward(self, x):
            return self.d(x.mean(-1))

    with pytest.raises(IncompatibleMixError):
        TargetControlMix(N, mix_coeff_layer=MyLogitMix, use_maximum_target_control_logprob=True)


def test_check_helper_is_a_no_op_when_ranges_agree():
    check_mix_compatibility(PooledProbMixCoeff(N), 'unit', 'test rule')


def test_mix_coeff_keeps_the_batch_axis_at_batch_size_one():
    """PooledLogitMixCoeff used a bare torch.squeeze, which drops the batch axis when B=1.

    The predicted tracks were unaffected because the subsequent unsqueeze broadcast
    correctly either way, so this surfaced only as `IndexError: too many indices` in
    caller code that did out['mix_coeff'][0, task]. Shapes must not depend on batch size.
    """
    head = layers.TargetControlMix(
        num_tasks=5,
        mix_coeff_layer=layers.PooledLogitMixCoeff,
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
    head = layers.TargetControlMix(
        num_tasks=3,
        mix_coeff_layer=layers.PooledLogitMixCoeff,
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


# --- the penalty acts on the target weight, not the raw layer output -----------------


def _head(mix_layer, use_max, factor=5.0, num_tasks=2):
    return layers.TargetControlMix(
        num_tasks=num_tasks,
        mix_coeff_layer=mix_layer,
        use_maximum_target_control_logprob=use_max,
        penalty_layer=functools.partial(layers.MixCoeffPenalty, factor=factor),
    )


def test_penalty_on_a_logit_head_sees_the_sigmoid_not_the_logit():
    """Regression for commit 0e247a9.

    PooledLogitMixCoeff emits an unbounded logit. Penalising it directly is unbounded below
    with a constant gradient, so it never stops pushing w toward 0. Penalising w is
    bounded in [0, factor]. Both released families were trained against w.
    """
    torch.manual_seed(0)
    head = _head(layers.PooledLogitMixCoeff, use_max=False, factor=5.0)
    x = torch.randn(4, 16, 32)
    head(x)
    head.eval()

    with torch.no_grad():
        out = head(x)

    w = torch.sigmoid(out['mix_coeff'])
    assert torch.allclose(out['target_weight'], w, atol=0), 'target_weight must be sigmoid(logit) here'

    penalty = out['penalty_loss']
    assert (penalty >= 0).all(), 'penalising w cannot go negative'
    assert (penalty <= 5.0).all(), 'penalising w is bounded by the factor'
    assert torch.allclose(penalty.squeeze(-1), w * 5.0, atol=1e-6)


def test_penalty_on_a_unit_head_sees_the_coefficient_unchanged():
    """PooledProbMixCoeff already ends in a sigmoid, so its output IS w and must not be squashed again."""
    torch.manual_seed(0)
    head = _head(layers.PooledProbMixCoeff, use_max=True, factor=3.0)
    x = torch.randn(4, 16, 32)
    head(x)
    head.eval()
    with torch.no_grad():
        out = head(x)

    assert torch.equal(out['target_weight'], out['mix_coeff'])
    assert torch.allclose(out['penalty_loss'].squeeze(-1), out['mix_coeff'] * 3.0, atol=1e-6)


def test_penalty_gradient_vanishes_as_the_control_track_takes_over():
    """The property that makes the historical penalty self-limiting.

    d/dm [sigmoid(m) * factor] = factor * w * (1 - w), which goes to 0 as w goes to 0.
    Penalising the logit instead gives a constant gradient of `factor` everywhere, so it
    keeps pushing without limit. This pins the self-limiting behaviour.
    """
    factor = 5.0
    gradients = []
    for logit in (-12.0, -6.0, 0.0):
        m = torch.tensor(logit, requires_grad=True)
        penalty = torch.sigmoid(m) * factor
        (grad,) = torch.autograd.grad(penalty, m)
        gradients.append(grad.item())

    assert gradients[0] < gradients[1] < gradients[2], 'gradient must shrink as w shrinks'
    assert gradients[0] < 1e-3, 'gradient must essentially vanish deep in the control regime'
    assert abs(gradients[2] - factor / 4) < 1e-6, 'gradient peaks at factor/4 when w = 0.5'


def test_penalty_requires_a_declared_range():
    """A hand-injected mix layer with no declared range cannot define a target weight."""
    head = _head(layers.PooledLogitMixCoeff, use_max=False)
    x = torch.randn(2, 16, 32)
    head(x)
    head.mix_coeff = layers.LambdaLayer(lambda _: torch.zeros(2, 2))
    with pytest.raises(layers.IncompatibleMixError, match='declares no mix_coeff_range'):
        head(x)


def test_no_declared_range_without_a_penalty_still_works():
    """Injecting a fixed coefficient is a legitimate thing to do in a test or a demo."""
    head = layers.TargetControlMix(
        num_tasks=2,
        mix_coeff_layer=layers.PooledLogitMixCoeff,
        use_maximum_target_control_logprob=False,
    )
    x = torch.randn(2, 16, 32)
    head(x)
    head.mix_coeff = layers.LambdaLayer(lambda _: torch.zeros(2, 2))
    out = head(x)
    assert 'penalty_loss' not in out
    assert 'target_weight' not in out, 'w is not derivable here, so it must not be invented'
