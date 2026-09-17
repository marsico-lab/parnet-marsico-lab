import logging
import sys

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


@gin.configurable()
class SequenceLinearMix(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()

        self.gloabel_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.LazyLinear(num_tasks)

    def forward(self, inputs):
        # inputs should have shape [batch, hidden_dim, length]

        x = torch.squeeze(self.gloabel_avg_pool(inputs))  # --> [batch, hidden_dim]
        logging.debug(f'x=torch.squeeze(self.gloabel_avg_pool(inputs)): {x.shape}')

        x = self.dense(x)  # --> [batch, num_tasks]
        logging.debug(f'self.dense(x): {x.shape}')

        return x


@gin.configurable()
class MixCoeffMLP(nn.Module):
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
class NewMixCoeffMLP(nn.Module):
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
class MixCoeffPenalty(nn.Module):
    """Simple mixing coefficient penalty that scales the mixing coefficient by a factor.

    This encourages the model to use lower the mixing coefficient, i.e. to increase the contribution of the control
    tack when combining the target and control tracks.
    """

    def __init__(self, factor=1.0) -> None:
        super().__init__()
        self.factor = factor

    def __call__(self, track_target, track_control, mix_coeff):
        # (B, num_tasks) -> (B, num_tasks)
        return mix_coeff * self.factor




@gin.configurable()
class AdditiveMix(nn.Module):
    """Additive mixing of target and control tracks.

    This layer takes a nucleotide-wise embedding, projects it to logits for the target and control tracks for a
    given number of tasks, computes a mixing coefficient for each task, and combines the target and control logits
    using the mixing coefficient.
    """

    def __init__(
        self,
        num_tasks,
        head_layer=LinearProjectionHead,
        mix_coeff_layer=MixCoeffMLP,
        penalty_layer=None,
        control_nograd=False,
        use_maximum_target_control_logprob=False,

    ):
        """Initializes AdditiveMix layer.

        Args:
            num_tasks (int): Number of tasks (i.e. eCLIP tracks).
            head_layer (nn.Module, optional): Layer to use for the output head. Defaults to LinearProjectionHead.
            mix_coeff_layer (nn.Module, optional): Layer to use for predicting mixing coefficients. Defaults to MixCoeffMLP.
            penalty_layer (nn.Module, optional): Layer to compute additional penalty added to the loss. Defaults to MixCoeffPenalty.
        """
        super().__init__()
        self.head_target = head_layer(num_tasks)
        self.head_control = head_layer(num_tasks)
        self.mix_coeff = mix_coeff_layer(num_tasks)
        self.penalty = penalty_layer() if penalty_layer is not None else None
        self.control_nograd = control_nograd
        self.use_maximum_target_control_logprob = use_maximum_target_control_logprob

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
            total_logprob = torch.where(
                is_both_neg_inf, torch.full_like(total_logprob, float('-inf')), total_logprob
            )
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
            'mix_coeff': mix_coeff.squeeze(-1),  # (B, num_tasks)
        }

        if self.penalty is not None:
            # add penalty loss (used only during training)
            return_dict['penalty_loss'] = self.penalty(target_logprob, control_logprob, mix_coeff)

        return return_dict


@gin.configurable()
class NewMixCoeffPenalty(nn.Module):
    def __init__(self, factor=1.0) -> None:
        super().__init__()
        self.factor = factor

    def __call__(self, track_target, track_control, mix_coeff):
        return mix_coeff * self.factor


@gin.configurable()
class NewAdditiveMix(nn.Module):
    def __init__(
        self,
        num_tasks,
        head_layer=LinearProjection,
        mix_coeff_layer=NewMixCoeffMLP,
        penalty_layer=None,
        control_nograd=False,
    ):
        super().__init__()

        self.head_target = head_layer(num_tasks)
        self.head_control = head_layer(num_tasks)
        self.mix_coeff = mix_coeff_layer(num_tasks)
        self.penalty = penalty_layer
        self.control_nograd = control_nograd

    def forward(self, inputs, **kwargs):
        # inputs should have shape [batch, hidden_dim, length]

        # project input feature map to logits for target and control --> [batch, num_tasks, length]
        target_logit = self.head_target(inputs)
        control_logit = self.head_control(inputs)

        # compute mixing coefficients --> [batch, num_tasks]
        mix_coeff = self.mix_coeff(inputs)
        mix_coeff = torch.unsqueeze(mix_coeff, dim=-1)

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
            'mix_coeff': mix_coeff.squeeze(-1),  # TODO: Add this.
        }

        if self.penalty is not None:
            return_dict['penalty_loss'] = self.penalty(target_logprob, control_logprob, mix_coeff)

        return return_dict
