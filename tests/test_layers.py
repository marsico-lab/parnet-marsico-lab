import torch

from parnet.layers import AdditiveMix, LambdaLayer, MixCoeffPenalty, NewAdditiveMix


def test_NewAdditiveMix():
    # Prepare mix-coeff layer. Provide a constant mixing coefficient and later overwrite the target/control head layers,
    # so that we can return fixed logit-tensors.
    layer = NewAdditiveMix(num_tasks=2)

    # Prepare inputs. We provide log-probs as logits, as they are easily generated from probs.
    layer.head_target = LambdaLayer(lambda _: torch.tensor([[0.1, 0.5, 0.4], [0.2, 0.3, 0.5]]).log().unsqueeze(0))
    layer.head_control = LambdaLayer(lambda _: torch.tensor([[0.2, 0.4, 0.4], [0.1, 0.8, 0.1]]).log().unsqueeze(0))
    layer.mix_coeff = LambdaLayer(lambda _: torch.tensor([0.5, 0.5]).unsqueeze(0))

    total_logprob = layer.forward(None)['total'].softmax(dim=-1)

    assert total_logprob.shape == (1, 2, 3), f'Expected shape (1, 2, 3), got {total_logprob.shape}'
    assert torch.allclose(total_logprob, torch.tensor([[0.15, 0.45, 0.40], [0.15, 0.55, 0.3]]).unsqueeze(0))


def _additive_mix_with_fixed_logits(target_probs, control_probs, mix_coeffs, use_maximum_target_control_logprob):
    num_tasks = target_probs.shape[0]
    layer = AdditiveMix(num_tasks=num_tasks, use_maximum_target_control_logprob=use_maximum_target_control_logprob)
    layer.head_target = LambdaLayer(lambda _: target_probs.log().unsqueeze(0))
    layer.head_control = LambdaLayer(lambda _: control_probs.log().unsqueeze(0))
    layer.mix_coeff = LambdaLayer(lambda _: mix_coeffs.unsqueeze(0))
    return layer


def test_AdditiveMix_max_logprob_matches_linear_mixing_reference():
    # use_maximum_target_control_logprob=True mixes target/control PROBABILITIES
    # linearly (mix_coeff * p_target + (1 - mix_coeff) * p_control), unlike the default
    # branch (which mixes additively in log-space) -- for ordinary, non-degenerate
    # inputs, the layer's log-space max-trick implementation should match this direct
    # probability-space computation.
    target_probs = torch.tensor([[0.1, 0.5, 0.4], [0.2, 0.3, 0.5]])
    control_probs = torch.tensor([[0.2, 0.4, 0.4], [0.1, 0.8, 0.1]])
    mix_coeffs = torch.tensor([0.5, 0.5])

    layer = _additive_mix_with_fixed_logits(target_probs, control_probs, mix_coeffs, True)
    total_prob = layer.forward(None)['total'].softmax(dim=-1)

    expected = (mix_coeffs.unsqueeze(-1) * target_probs + (1 - mix_coeffs.unsqueeze(-1)) * control_probs).unsqueeze(0)
    assert torch.allclose(total_prob, expected, atol=1e-4)


def test_AdditiveMix_max_logprob_handles_double_neg_inf_without_nan():
    # Regression test for the "Logits contain NaN or Inf values." crash. In production
    # this -inf comes from log_softmax(target_logit)/(control_logit) underflowing at a
    # position far from a confidently-predicted peak (more likely in float16, and more
    # likely under a high penalty that sharpens predictions) -- reproduced here directly
    # by feeding a -inf logit at one position, which log_softmax carries straight
    # through as -inf regardless of dtype, so the test is deterministic and
    # dtype-independent.
    target_logit = torch.tensor([[0.0, 1.0, float('-inf')]])  # (num_tasks=1, L=3)
    control_logit = torch.tensor([[0.5, -0.5, float('-inf')]])

    layer = AdditiveMix(num_tasks=1, use_maximum_target_control_logprob=True)
    layer.head_target = LambdaLayer(lambda _: target_logit.unsqueeze(0))  # -> (B=1, num_tasks=1, L=3)
    layer.head_control = LambdaLayer(lambda _: control_logit.unsqueeze(0))
    layer.mix_coeff = LambdaLayer(lambda _: torch.tensor([0.5]).unsqueeze(0))

    output = layer.forward(None)  # must not raise ValueError('Logits contain NaN or Inf values.')

    assert not torch.isnan(output['total']).any()
    assert torch.isneginf(output['total'][0, 0, 2]), "the double -inf position should be exactly -inf, not NaN"
    # The two ordinary positions should be finite, unaffected by the fix.
    assert torch.isfinite(output['total'][0, 0, 0])
    assert torch.isfinite(output['total'][0, 0, 1])


def test_MixCoeffPenalty_default_lambda_anchor_is_a_no_op():
    # lambda_anchor defaults to 0.0 -- must reproduce the original factor-only behavior
    # exactly, so no existing checkpoint's training run is affected retroactively.
    penalty = MixCoeffPenalty(factor=2.0)
    mix_coeff = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(penalty(None, None, mix_coeff), mix_coeff * 2.0)


def test_MixCoeffPenalty_lambda_anchor_penalizes_saturation_symmetrically():
    # The anchor term is (mix_coeff - 0.5)**2, scaled by lambda_anchor -- zero at 0.5,
    # equal and maximal at both mix_coeff=0.0 and mix_coeff=1.0 (symmetric, unlike the
    # directional `factor` term which only discourages high mix_coeff).
    penalty = MixCoeffPenalty(factor=0.0, lambda_anchor=1.0)
    mix_coeff = torch.tensor([0.0, 0.5, 1.0])
    output = penalty(None, None, mix_coeff)
    expected = torch.tensor([0.25, 0.0, 0.25])
    assert torch.allclose(output, expected)
