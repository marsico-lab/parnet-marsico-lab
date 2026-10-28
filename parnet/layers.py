import logging
import sys
import warnings

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(torch.nn.Module):
    def __init__(self, fun):
        super(LambdaLayer, self).__init__()
        self.fun = fun

    def forward(self, *args, **kwargs):
        return self.fun(*args, **kwargs)


@gin.configurable()
class StemConv1D(nn.Module):
    """Class to be used as first layer of a model.

    Applies an ordenary 1D convolution, followed by  batch normalization, activation and dropout. Seperating this
    layer from the rest of the model allows us to inject hyperparameters for the first layer only via gin-config.
    """

    def __init__(self, filters=128, kernel_size=12, activation=nn.ReLU(), dropout=None):
        """Initializes StemConv1D.

        Args:
            filters (int, optional): Number of convolutional kernels/filters. Defaults to 128.
            kernel_size (int, optional): Size of kernels/filters. Defaults to 12.
            activation (Any, optional): Activation function. Defaults to nn.ReLU().
            dropout (float, optional): Dropout probability. Defaults to None, in which case no dropout will be applied.
        """
        super().__init__()

        self.conv1d = nn.LazyConv1d(filters, kernel_size, padding='same')
        self.batch_norm = nn.BatchNorm1d(filters)
        self.act = activation
        self.dropout = nn.Dropout1d(dropout) if dropout is not None else None

    def forward(self, x, **kwargs):
        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


@gin.configurable()
class ResConvBlock1D(nn.Module):
    # TODO: Add documentation.
    def __init__(
        self,
        filters=128,
        kernel_size=3,
        dropout=0.25,
        activation=nn.ReLU(),
        dilation=1,
        residual=True,
    ):
        super().__init__()

        self.conv1d = nn.LazyConv1d(filters, kernel_size=kernel_size, dilation=int(dilation), padding='same')
        self.batch_norm = nn.BatchNorm1d(filters)
        self.act = activation
        self.dropout = nn.Dropout1d(dropout) if dropout is not None else None
        self.residual = residual

    def forward(self, inputs, **kwargs):
        x = inputs

        try:
            x = self.conv1d(x)
        except:
            print(x.shape, x.dtype, file=sys.stderr)
            raise

        x = self.batch_norm(x)
        x = self.act(x)
        # dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # residual
        if self.residual:
            x = inputs + x

        return x


@gin.configurable()
class LinearProjectionHead(nn.Module):
    """Performs a linear projection of the nucleotide-wise embeddings to logits for each task.

    With the default kernel_size=1, this is pointwise (each output position depends only on its
    own input position). A larger kernel_size lets each output position also draw on neighboring
    positions, matching the original RBPNet ProfileHead's local-window behavior.
    """

    def __init__(self, num_tasks=None, kernel_size=1) -> None:
        super().__init__()
        if num_tasks is None:
            raise ValueError('num_tasks must be specified')
        self.pointwise = nn.LazyConv1d(num_tasks, kernel_size=kernel_size, bias=False, padding='same')

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # (B, hidden_dim, L) --> (B, num_tasks, L)
        return self.pointwise(inputs)


@gin.configurable()
class LinearProjection(nn.Module):
    """Performs a linear projection of the input to the specified number of output channels.

    To be used as an upsampling/downsampling layer. See LinearProjectionHead's docstring
    regarding kernel_size.
    """

    def __init__(self, out_features=128, kernel_size=1) -> None:
        super().__init__()
        self.pointwise_conv = nn.LazyConv1d(out_features, kernel_size=kernel_size, bias=False, padding='same')

    def forward(self, x):
        return self.pointwise_conv(x)


class IncompatibleMixError(TypeError):
    """Raised when a mix-coefficient layer's output range does not match the mixing rule.

    A mixing rule consumes ``mix_coeff`` as one of two different quantities, and pairing it
    with a layer that emits the other produces silently wrong numbers or NaN rather than an
    error at the point of the mistake. The two ranges are:

    ``"logit"``
        An unbounded real number, added to the target log-probability in log space, so the
        effective target weight is ``sigmoid(mix_coeff)``. Emitted by
        :class:`PooledLogitMixCoeff`; consumed by the log-sum branch of :class:`TargetControlMix`
        (``use_maximum_target_control_logprob=False``).

    ``"unit"``
        A probability in [0, 1], used directly as the weight in a convex combination
        ``w * exp(target) + (1 - w) * exp(control)``. Emitted by :class:`PooledProbMixCoeff` and
        :class:`PooledProbMixCoeffV2`, which end in a sigmoid; consumed by
        :class:`ConvexTargetControlMix` and by the max-logprob branch of :class:`TargetControlMix`.

    The two failure modes this prevents, both observed in practice:

    - ``"logit"`` into a convex rule: ``(1 - w)`` becomes large and negative, so the
      logarithm receives a negative argument wherever the control track dominates at that
      position, and the result is NaN. Measured on the released 21M, whose ``mix_coeff``
      spans -4.6 to 3132: 96.6 per cent of values fall outside [0, 1] and 26.8 per cent of
      positions give a negative argument.
    - ``"unit"`` into the log-sum rule: a value already squashed by a sigmoid is treated as
      a log-space weight, so the effective target weight becomes
      ``sigmoid(sigmoid(logit))``, confined to (0.5, 0.731). The control track can then
      never fall below 26.9 per cent and the penalty saturates. This is what
      :class:`TargetControlMix`'s own default arguments produce.
    """


#: Output range declared by each known mix-coefficient layer.
MIX_COEFF_RANGES = {
    'PooledLogitMixCoeff': 'logit',
    'PooledProbMixCoeff': 'unit',
    'PooledProbMixCoeffV2': 'unit',
}


def mix_coeff_range(layer):
    """Return the declared output range of a mix-coefficient layer, or None if unknown."""
    declared = getattr(layer, 'mix_coeff_range', None)
    if declared is not None:
        return declared
    return MIX_COEFF_RANGES.get(type(layer).__name__)


def check_mix_compatibility(mix_layer, required, rule, *, strict=True):
    """Raise :class:`IncompatibleMixError` if ``mix_layer`` does not emit ``required``.

    An unrecognised layer passes unchecked, because a user-supplied layer may legitimately
    declare nothing; set ``mix_coeff_range`` on it to opt in.
    """
    got = mix_coeff_range(mix_layer)
    if got is None or got == required:
        return
    if required == 'unit':
        why = (
            'A logit fed to a convex combination makes (1 - w) negative, so the logarithm '
            'receives a negative argument and the result is NaN.'
        )
    else:
        why = (
            'A sigmoid output fed to the log-sum rule is squashed twice, confining the '
            'effective target weight to (0.5, 0.731).'
        )
    msg = (
        f'{type(mix_layer).__name__} emits mix_coeff in range {got!r}, but {rule} requires '
        f'{required!r}. {why} Pass a matching mix_coeff_layer, or strict_mix=False to '
        'proceed anyway.'
    )
    if strict:
        raise IncompatibleMixError(msg)
    warnings.warn(msg, UserWarning, stacklevel=3)


#: Attributes that were added to layer classes after some released checkpoints were written.
#: ``torch.load`` on a whole pickled module restores ``__dict__`` without calling
#: ``__init__``, so an older checkpoint simply lacks them and ``forward`` raises
#: ``AttributeError`` on the first access. Each default below is the behaviour that was in
#: force *before* the attribute existed, so filling it in reproduces the checkpoint's
#: original semantics rather than imposing today's default.
#:
#: Three have been hit in practice on released files:
#:   MixCoeffPenalty.anchor_strength                      (3M and 7M checkpoints)
#:   TargetControlMix.use_maximum_target_control_logprob     (released 21M)
#:   TargetControlMix.control_nograd                         (released 21M)
_LEGACY_ATTRIBUTE_DEFAULTS = {
    'TargetControlMix': {
        # The log-sum branch was the only behaviour before the flag existed.
        'use_maximum_target_control_logprob': False,
        # Gradients flowed through the control track before this switch was added.
        'control_nograd': False,
    },
    'ConvexTargetControlMix': {
        'control_nograd': False,
    },
    'RBPNet': {
        # Three different representations of "no projection" exist in released files:
        #   released 21M        attribute ABSENT entirely
        #   historical 7M       present, value None
        #   3M backbones        present, value nn.Identity
        # None is the canonical "there is none"; forward guards on it, so all three behave
        # identically once this default is filled in.
        'projection': None,
    },
    'PooledLogitMixCoeff': {
        # Added 2026-10. Every released checkpoint predates it and used no activation.
        'activation': None,
    },
    'MixCoeffPenalty': {
        # Documented as a no-op default matching prior behaviour.
        'anchor_strength': 0.0,
    },
    'NewMixCoeffPenalty': {
        'anchor_strength': 0.0,
    },
}


class _LegacyStateMixin:
    """Fill in attributes added after a checkpoint was pickled.

    Without this, every attribute added to a layer class breaks every checkpoint written
    before it, and the break surfaces as an ``AttributeError`` from deep inside ``forward``
    rather than at load time. Callers previously worked around it with an external table of
    per-model patches; the class itself is the right place for the knowledge.
    """

    def __setstate__(self, state):
        if isinstance(state, dict) and 'state' not in state:
            self.__dict__.update(state)
        else:  # torch's own (state, slots) form
            super().__setstate__(state)
        defaults = _LEGACY_ATTRIBUTE_DEFAULTS.get(type(self).__name__, {})
        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value)


def mix_coeff_to_target_weight(mix_coeff, mix_layer):
    """Convert a mixing layer's raw output into the target weight w in [0, 1].

    w is the quantity with a meaning: the share of the prediction carried by the target
    track. Whether a layer's output already IS w depends on its ``mix_coeff_range``. A
    ``'unit'`` layer ends in a sigmoid and emits w directly; a ``'logit'`` layer emits an
    unbounded score whose target weight is its sigmoid.

    Callers have been doing this by hand and getting it wrong in both directions: applying
    a sigmoid to an already-bounded 7M coefficient, which compresses every value toward
    0.5, and omitting it for the 21M. Doing it once, from the layer's own declared range,
    removes the choice.

    Returns ``None`` when the layer declares no range, which is the case for a
    hand-injected stand-in such as :class:`LambdaLayer`. Only the penalty genuinely needs
    w, and it raises on ``None``; everything else carries on.
    """
    range_ = getattr(type(mix_layer), 'mix_coeff_range', None)
    if range_ == 'unit':
        return mix_coeff
    if range_ == 'logit':
        return torch.sigmoid(mix_coeff)
    return None


def _require_target_weight(target_weight, mix_layer):
    """Return *target_weight*, or explain why it could not be derived."""
    if target_weight is None:
        raise IncompatibleMixError(
            f'{type(mix_layer).__name__} declares no mix_coeff_range, so its output '
            'cannot be interpreted as a target weight, and the penalty acts on the '
            'target weight. Set mix_coeff_range to "unit" or "logit" on the layer.'
        )
    return target_weight


@gin.configurable()
def _mix_coeff_range_for(activation) -> str:
    """Which coefficient range an output activation produces.

    Only the cases we can state are allowed. Guessing would feed a wrong range to the
    compatibility guard, which is the one thing that guard exists to prevent.
    """
    if activation is None:
        return 'logit'
    if isinstance(activation, nn.Sigmoid):
        return 'unit'
    raise IncompatibleMixError(
        f'cannot infer the output range of {type(activation).__name__}. Pass None for a raw '
        'logit or nn.Sigmoid() for a unit-range coefficient; any other activation needs its '
        'range stated here before it can be used as a mix coefficient.'
    )


class PooledLogitMixCoeff(nn.Module):
    #: Range when no activation is given. Instances override it, because the range follows
    #: the activation. Both are recorded in the config: an activation that is not written
    #: down gets rebuilt as the default, which is how the 3M backbones were once rebuilt
    #: with ReLU where they had been trained with GELU.
    mix_coeff_range = 'logit'

    def __init__(self, num_tasks, activation=None):
        super().__init__()

        self.gloabel_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.LazyLinear(num_tasks)
        # Without this the layer welds depth to range: a bounded coefficient could only be
        # had by moving to the two-layer MLP. None reproduces the released 21M exactly.
        self.activation = activation
        self.mix_coeff_range = _mix_coeff_range_for(activation)

    def forward(self, inputs):
        # inputs should have shape [batch, hidden_dim, length]

        # squeeze the pooled length axis ONLY. A bare torch.squeeze() drops every size-1
        # dimension, which at batch size 1 removes the batch axis too and returns
        # mix_coeff with shape (num_tasks,) instead of (1, num_tasks). The totals were
        # unaffected because the later unsqueeze broadcast correctly either way, so this
        # surfaced only as an indexing error in caller code.
        x = torch.squeeze(self.gloabel_avg_pool(inputs), dim=-1)  # --> [batch, hidden_dim]
        logging.debug(f'x=torch.squeeze(self.gloabel_avg_pool(inputs)): {x.shape}')

        x = self.dense(x)  # --> [batch, num_tasks]
        logging.debug(f'self.dense(x): {x.shape}')

        # getattr, not self.activation: checkpoints pickled before 2026-10 have no such
        # attribute, and __setstate__ fills it only for classes listed in the legacy table.
        activation = getattr(self, 'activation', None)
        return x if activation is None else activation(x)


@gin.configurable()
class PooledProbMixCoeff(nn.Module):
    #: Output range of ``forward``; see :class:`IncompatibleMixError`.
    mix_coeff_range = 'unit'

    """2-layer MLP that takes a feature map as input and outputs a mixing coefficient for each task."""

    def __init__(self, num_tasks, units=128, act=nn.ReLU()) -> None:
        super().__init__()

        self.dense1 = nn.LazyLinear(units)
        self.act = act
        self.dense2 = nn.LazyLinear(num_tasks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # (B, hidden_dim, L) --> (B, hidden_dim)
        x = inputs.mean(-1)  # --> [batch, hidden_dim]
        logging.debug(f'{x.shape}')

        # (B, hidden_dim) --> (B, units)
        x = self.dense1(x)  # --> [batch, units]
        x = self.act(x)
        logging.debug(f'self.dense(x): {x.shape}')

        x = self.dense2(x)  # --> [batch, num_tasks]

        return F.sigmoid(x)


@gin.configurable()
class PooledProbMixCoeffV2(nn.Module):
    #: Output range of ``forward``; see :class:`IncompatibleMixError`.
    mix_coeff_range = 'unit'

    def __init__(self, num_tasks, units=128, act=nn.ReLU()) -> None:
        super().__init__()

        self.dense1 = nn.LazyLinear(units)
        self.act = act
        self.dense2 = nn.LazyLinear(num_tasks)

    def forward(self, x):
        # x should have shape [batch, hidden_dim, length]

        x = x.mean(-1)  # --> [batch, hidden_dim]
        logging.debug(f'x (pooled): {x.shape}')

        x = self.dense1(x)  # --> [batch, units]
        x = self.act(x)
        logging.debug(f'self.dense(x): {x.shape}')

        x = self.dense2(x)  # --> [batch, num_tasks]

        return F.sigmoid(x)


@gin.configurable()
class MixCoeffPenalty(_LegacyStateMixin, nn.Module):
    """Mixing coefficient penalty: a directional term plus an optional anti-saturation anchor.

    `factor` encourages a lower mixing coefficient, i.e. more contribution from the control
    track -- this is the original, directional penalty, and is a no-op at `factor=0`.

    `anchor_strength` is a separate, non-directional term: `anchor_strength * (mix_coeff - 0.5)**2`,
    always active regardless of `factor`, pulling mix_coeff away from full saturation at 0 or 1
    in either direction. Added to address mix_coeff collapsing to the extremes even at
    `factor=0`, observed in head-only fine-tunes with `use_maximum_target_control_logprob=True`
    (nothing else constrains mix_coeff there). Defaults to 0.0 (no-op, matches prior behavior).
    """

    def __init__(self, factor=1.0, anchor_strength=0.0) -> None:
        super().__init__()
        self.factor = factor
        self.anchor_strength = anchor_strength

    def __call__(self, track_target, track_control, mix_coeff):
        # (B, num_tasks) -> (B, num_tasks)
        return mix_coeff * self.factor + self.anchor_strength * (mix_coeff - 0.5) ** 2


@gin.configurable()
class TargetControlMix(_LegacyStateMixin, nn.Module):
    """Additive mixing of target and control tracks.

    This layer takes a nucleotide-wise embedding, projects it to logits for the target and control tracks for a
    given number of tasks, computes a mixing coefficient for each task, and combines the target and control logits
    using the mixing coefficient.
    """

    def __init__(
        self,
        num_tasks,
        head_layer=LinearProjectionHead,
        mix_coeff_layer=PooledProbMixCoeff,
        penalty_layer=None,
        control_nograd=False,
        use_maximum_target_control_logprob=False,
        strict_mix=True,
    ):
        """Initializes TargetControlMix layer.

        Args:
            num_tasks (int): Number of tasks (i.e. eCLIP tracks).
            head_layer (nn.Module, optional): Layer to use for the output head. Defaults to LinearProjectionHead.
            mix_coeff_layer (nn.Module, optional): Layer to use for predicting mixing coefficients. Defaults to PooledProbMixCoeff.
            penalty_layer (nn.Module, optional): Layer to compute additional penalty added to the loss. Defaults to MixCoeffPenalty.
        """
        super().__init__()
        self.head_target = head_layer(num_tasks)
        self.head_control = head_layer(num_tasks)
        self.mix_coeff = mix_coeff_layer(num_tasks)
        self.penalty = penalty_layer() if penalty_layer is not None else None
        self.control_nograd = control_nograd
        self.use_maximum_target_control_logprob = use_maximum_target_control_logprob
        # Which quantity this instance consumes depends on the branch it will take: the
        # max-logprob branch is a convex combination and needs a weight in [0, 1]; the
        # default log-sum branch adds mix_coeff in log space and needs a logit.
        check_mix_compatibility(
            self.mix_coeff,
            'unit' if use_maximum_target_control_logprob else 'logit',
            f'TargetControlMix with use_maximum_target_control_logprob={use_maximum_target_control_logprob}',
            strict=strict_mix,
        )

    def forward(self, inputs, **kwargs):
        # inputs: (B, hidden_dim, L)

        # project input feature map to logits for target and control tracks
        # (B, hidden_dim, L) -> (B, num_tasks, L)
        target_logit = self.head_target(inputs)
        control_logit = self.head_control(inputs)
        logging.debug(f'{target_logit.shape=}, {control_logit.shape=}')

        # compute mixing coefficients for each task
        # (B, hidden_dim, L) -> (B, num_tasks, 1)
        mix_coeff = self.mix_coeff(inputs)
        mix_coeff = torch.unsqueeze(mix_coeff, dim=-1)
        target_weight = mix_coeff_to_target_weight(mix_coeff, self.mix_coeff)
        logging.debug(f'{mix_coeff.shape=}')

        # Additive mixing of target and control tracks, weighted by the mixing coefficient. The logsumexp trick
        # is used to avoid numerical instability.
        target_logprob = target_logit - torch.logsumexp(target_logit, dim=-1, keepdim=True)
        control_logprob = control_logit - torch.logsumexp(control_logit, dim=-1, keepdim=True)

        if self.use_maximum_target_control_logprob:
            max_logprob = torch.maximum(target_logprob, control_logprob)
            # Guard against -inf - -inf = NaN: this happens when BOTH target_logprob
            # and control_logprob underflow to exactly -inf at some position (e.g. in
            # float16, far from a confidently-predicted peak, under a strongly-peaked
            # distribution -- more likely at high penalty values). The correct limiting
            # value there is -inf (both probabilities are ~0, so their weighted sum is
            # too), not NaN. Substitute a finite stand-in only for computing the
            # (otherwise-unstable) subtraction below, then force the true -inf result
            # back in afterwards -- every other position is completely unaffected,
            # since `is_both_neg_inf` is False there and `safe_max` equals `max_logprob`
            # exactly.
            is_both_neg_inf = torch.isneginf(max_logprob)
            safe_max = torch.where(is_both_neg_inf, torch.zeros_like(max_logprob), max_logprob)
            total_logprob = safe_max + torch.log(
                mix_coeff * torch.exp(target_logprob - safe_max)
                + (1 - mix_coeff) * torch.exp(control_logprob - safe_max)
                + 1e-10  # small constant to avoid numerical issues
            )
            total_logprob = torch.where(is_both_neg_inf, torch.full_like(total_logprob, float('-inf')), total_logprob)
        else:
            total_logprob = torch.logsumexp(
                torch.stack(
                    [
                        mix_coeff + target_logprob,
                        (control_logprob.detach() if self.control_nograd else control_logprob),
                    ],
                    dim=0,
                ),
                dim=0,
            )

        # check for NaN/+Inf in total_logprob. -inf is legitimate here: it means a
        # position's probability is exactly 0 (e.g. both target and control underflowed
        # to -inf there). Only NaN and +inf indicate an actual numerical bug.
        if torch.isnan(total_logprob).any() or torch.isposinf(total_logprob).any():
            print(
                f'{target_logprob.min()}, {target_logprob.max()}, {target_logprob.mean()}, ',
                file=sys.stderr,
                flush=True,
            )
            print(
                f'{control_logprob.min()}, {control_logprob.max()}, {control_logprob.mean()}, ',
                file=sys.stderr,
                flush=True,
            )
            print(f'{total_logprob=}, {mix_coeff=}, {target_logprob=}, {control_logprob=}', file=sys.stderr, flush=True)
            raise ValueError('Logits contain NaN or Inf values.')

        return_dict = {
            'target': target_logprob,  # (B, num_tasks, L)
            'control': control_logprob,  # (B, num_tasks, L)
            'total': total_logprob,  # (B, num_tasks, L)
            'mix_coeff': mix_coeff.squeeze(-1),
            **({'target_weight': target_weight.squeeze(-1)} if target_weight is not None else {}),  # (B, num_tasks)
        }

        if self.penalty is not None:
            # add penalty loss (used only during training)
            # The penalty acts on the TARGET WEIGHT, not on the raw layer output. For a
            # 'logit' layer the two are different objectives: penalising the logit is
            # unbounded below with a constant gradient, so it never stops pushing w
            # toward 0, while penalising w is bounded in [0, factor] with gradient
            # factor*w*(1-w), which vanishes once the control track dominates. Both
            # released families were trained against w. Penalising the logit arrived
            # with commit 0e247a9 and is the likely cause of w collapsing in epoch 0.
            return_dict['penalty_loss'] = self.penalty(
                target_logprob,
                control_logprob,
                _require_target_weight(target_weight, self.mix_coeff),
            )

        return return_dict


@gin.configurable()
class NewMixCoeffPenalty(_LegacyStateMixin, nn.Module):
    def __init__(self, factor=1.0) -> None:
        super().__init__()
        self.factor = factor

    def __call__(self, track_target, track_control, mix_coeff):
        return mix_coeff * self.factor


@gin.configurable()
class ConvexTargetControlMix(_LegacyStateMixin, nn.Module):
    def __init__(
        self,
        num_tasks,
        head_layer=LinearProjection,
        mix_coeff_layer=PooledProbMixCoeffV2,
        penalty_layer=None,
        control_nograd=False,
        strict_mix=True,
    ):
        super().__init__()

        self.head_target = head_layer(num_tasks)
        self.head_control = head_layer(num_tasks)
        self.mix_coeff = mix_coeff_layer(num_tasks)
        self.penalty = penalty_layer
        self.control_nograd = control_nograd
        # Always a convex combination, so always a weight in [0, 1].
        check_mix_compatibility(self.mix_coeff, 'unit', 'ConvexTargetControlMix', strict=strict_mix)

    def forward(self, inputs, **kwargs):
        # inputs should have shape [batch, hidden_dim, length]

        # project input feature map to logits for target and control --> [batch, num_tasks, length]
        target_logit = self.head_target(inputs)
        control_logit = self.head_control(inputs)

        # compute mixing coefficients --> [batch, num_tasks]
        mix_coeff = self.mix_coeff(inputs)
        mix_coeff = torch.unsqueeze(mix_coeff, dim=-1)
        target_weight = mix_coeff_to_target_weight(mix_coeff, self.mix_coeff)

        logging.debug(f'mix_coeff.shape: {mix_coeff.shape}')

        # additive mixing of target and control tracks with control track weigthed
        # by the mixing coefficient --> [batch, num_tasks, length]
        target_logprob = target_logit - torch.logsumexp(target_logit, dim=-1, keepdim=True)
        control_logprob = control_logit - torch.logsumexp(control_logit, dim=-1, keepdim=True)

        if self.control_nograd:
            control_logprob = control_logprob.detach()

        # use logsumexp trick to avoid numerical instability
        max_logprob = torch.maximum(target_logprob, control_logprob)
        total_logprob = max_logprob + torch.log(
            mix_coeff * torch.exp(target_logprob - max_logprob)
            + (1 - mix_coeff) * torch.exp(control_logprob - max_logprob)
        )

        return_dict = {
            'target': target_logprob,
            'control': control_logprob,
            'total': total_logprob,
            'mix_coeff': mix_coeff.squeeze(-1),
            **({'target_weight': target_weight.squeeze(-1)} if target_weight is not None else {}),  # TODO: Add this.
        }

        if self.penalty is not None:
            # The penalty acts on the TARGET WEIGHT, not on the raw layer output. For a
            # 'logit' layer the two are different objectives: penalising the logit is
            # unbounded below with a constant gradient, so it never stops pushing w
            # toward 0, while penalising w is bounded in [0, factor] with gradient
            # factor*w*(1-w), which vanishes once the control track dominates. Both
            # released families were trained against w. Penalising the logit arrived
            # with commit 0e247a9 and is the likely cause of w collapsing in epoch 0.
            return_dict['penalty_loss'] = self.penalty(
                target_logprob,
                control_logprob,
                _require_target_weight(target_weight, self.mix_coeff),
            )

        return return_dict


#: Names these classes had before the 2026-10 rename. The old names said how the layer was
#: built ("additive", "a linear on the sequence"); the new ones say what it computes and
#: what it emits, which is the distinction that actually bit us: the combination rule and
#: the coefficient's range are independent axes, and the old names conflated them.
#:
#: Resolution goes through ``__getattr__`` rather than plain assignments so that an old
#: name warns, while still resolving for ``torch.load`` on a pickled checkpoint, which
#: looks attributes up on the module in exactly this way.
_RENAMED_IN_2026_10 = {
    'AdditiveMix': 'TargetControlMix',
    'NewAdditiveMix': 'ConvexTargetControlMix',
    'SequenceLinearMix': 'PooledLogitMixCoeff',
    'MixCoeffMLP': 'PooledProbMixCoeff',
    'NewMixCoeffMLP': 'PooledProbMixCoeffV2',
}


def __getattr__(name: str):
    """Resolve a pre-rename class name, with a warning."""
    if name in _RENAMED_IN_2026_10:
        new = _RENAMED_IN_2026_10[name]
        warnings.warn(
            f'parnet.layers.{name} was renamed to {new} in 2026-10 and the old name will '
            f'be removed at 1.0. Behaviour is unchanged. Pickled checkpoints that still '
            f'name {name} keep loading; re-export them to drop the alias.',
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
