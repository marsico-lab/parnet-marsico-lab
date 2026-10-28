"""Save and load PARNET models as ``model.safetensors`` plus ``config.json``.

Whole-module pickles are deprecated. A pickle stores a class path and executes code on
load, so a renamed class or an attribute added after the file was written breaks it, and
the break surfaces far from its cause. Safetensors stores tensors and a flat metadata
dict and nothing else: it cannot name a class and cannot execute anything.

The architecture therefore moves into a sibling ``config.json``, which introduces the one
risk worth guarding: a config that disagrees with the weights. The rule here is that
**the weights are the single source of truth**. ``use_projection`` in the config is
derived metadata; :func:`from_pretrained` recomputes it from the state dict and raises if
the two disagree rather than silently trusting either.

    model.save_pretrained('models/parnet-3m-E2')
    model = RBPNet.from_pretrained('models/parnet-3m-E2')   # returns eval mode

Both functions take a STEM. ``.safetensors`` and ``.config.json`` are appended.
"""

from __future__ import annotations

import datetime
import functools
import json
import pathlib
from typing import Any

import torch.nn as nn
from pydantic import ValidationError

from parnet import layers as _layers
from parnet.config import ModelConfig

#: Classes that may be named in a ``config.json``. Resolution goes through this table and
#: never through ``getattr`` on a module, so a config file cannot cause an arbitrary
#: import. Adding a class here is a deliberate act.
CLASS_REGISTRY: dict[str, type] = {
    'StemConv1D': _layers.StemConv1D,
    'ResConvBlock1D': _layers.ResConvBlock1D,
    'LinearProjectionHead': _layers.LinearProjectionHead,
    'LinearProjection': _layers.LinearProjection,
    'PooledLogitMixCoeff': _layers.PooledLogitMixCoeff,
    'PooledProbMixCoeff': _layers.PooledProbMixCoeff,
    'PooledProbMixCoeffV2': _layers.PooledProbMixCoeffV2,
    'MixCoeffPenalty': _layers.MixCoeffPenalty,
    'NewMixCoeffPenalty': _layers.NewMixCoeffPenalty,
    'TargetControlMix': _layers.TargetControlMix,
    'ConvexTargetControlMix': _layers.ConvexTargetControlMix,
}

# Configs written before the 2026-10 rename name the old classes. They resolve to the
# same objects, so an already-exported model keeps loading without being rewritten.
# Reading one does NOT warn: a config is a file, not a call site, and nobody can act on
# a warning raised while loading someone else's checkpoint. Re-exporting rewrites them.
CLASS_REGISTRY.update({old: CLASS_REGISTRY[new] for old, new in _layers._RENAMED_IN_2026_10.items()})

#: Activation classes that may be named in a config. Activations are NOT parameters, so
#: ``load_state_dict`` restores weights perfectly while silently leaving the wrong
#: activation in place. That exact defect invalidated an earlier batch of 7M results and
#: made the first 3M export fail to reproduce its own reference outputs, so every
#: activation is recorded and restored explicitly.
ACTIVATION_REGISTRY: dict[str, type] = {
    'ReLU': nn.ReLU,
    'GELU': nn.GELU,
    'SiLU': nn.SiLU,
    'ELU': nn.ELU,
    'LeakyReLU': nn.LeakyReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'Identity': nn.Identity,
}

CONFIG_SUFFIX = '.config.json'
WEIGHTS_SUFFIX = '.safetensors'


class ConfigWeightMismatch(ValueError):
    """Raised when ``config.json`` claims something the weights contradict."""


def _mix_coeff_activation_name(mix) -> str | None:
    """Name of the mix-coefficient layer's output activation, under either spelling."""
    for attribute in ('act', 'activation'):
        value = getattr(mix, attribute, None)
        if value is not None:
            return type(value).__name__
    return None


def _resolve_activation(name: str | None) -> nn.Module | None:
    if name is None:
        return None
    try:
        return ACTIVATION_REGISTRY[name]()
    except KeyError:
        raise KeyError(
            f'{name!r} is not a registered activation. Known: {sorted(ACTIVATION_REGISTRY)}. '
            'Add it to parnet.pretrained.ACTIVATION_REGISTRY if it is genuinely used.'
        ) from None


def _resolve(name: str | None) -> type | None:
    if name is None:
        return None
    try:
        return CLASS_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f'{name!r} is not a registered PARNET class. Known: {sorted(CLASS_REGISTRY)}. '
            'If this is a new class, add it to parnet.pretrained.CLASS_REGISTRY.'
        ) from None


def state_dict_has_projection(state_dict: dict[str, Any]) -> bool:
    """True when the state dict contains weights for a top-level projection layer.

    This is the authoritative answer to "does this model have a projection". An
    ``nn.Identity`` projection has no parameters and is therefore indistinguishable from
    no projection at all in a state dict, which is precisely why ``None`` is the canonical
    representation of "there is none" and why this function looks for weights rather than
    for an attribute.
    """
    return any(key.startswith('projection.') and key.endswith(('.weight', '.bias')) for key in state_dict)


def _output_conv(head_layer: nn.Module) -> nn.Conv1d:
    """Return the single Conv1d inside an output-head layer, whatever it is called."""
    convs = [m for m in head_layer.modules() if isinstance(m, nn.Conv1d)]
    if len(convs) != 1:
        raise ValueError(
            f'expected exactly one Conv1d in {type(head_layer).__name__}, found {len(convs)}; '
            'parnet.pretrained.describe cannot infer n_tasks from it'
        )
    return convs[0]


def describe(model: nn.Module) -> dict[str, Any]:
    """Derive the ``architecture`` block of a config from a live model.

    Every value is read off the model, never passed in, so a config written here cannot
    disagree with the object it describes.
    """
    head = model.head
    mix = head.mix_coeff
    penalty = getattr(head, 'penalty', None)
    projection = getattr(model, 'projection', None)
    has_projection = any(True for _ in projection.parameters()) if projection is not None else False

    # The output conv is named `pointwise` on LinearProjectionHead and `pointwise_conv` on
    # LinearProjection. Find it by type rather than by attribute name, so a new head class
    # does not silently produce n_tasks=None in a config.
    conv = _output_conv(head.head_target)

    arch: dict[str, Any] = {
        'class': type(model).__name__,
        'n_body_blocks': len(model.body),
        # Width of the residual stream, set by the stem's filter count. Distinct from
        # embedding_dim: with no projection the head sees this width directly, with a
        # projection it sees the projection's output instead. Both are needed to rebuild.
        'body_width': int(model.stem.conv1d.weight.shape[0]),
        'embedding_dim': int(conv.weight.shape[1]),
        'use_projection': has_projection,
        # Everything below is non-parametric: it does not appear in the state dict, so a
        # config that omits it produces a model that loads cleanly and computes something
        # different. Recorded explicitly for that reason.
        'stem': {
            'class': type(model.stem).__name__,
            'kernel_size': int(model.stem.conv1d.kernel_size[0]),
            'activation': type(model.stem.act).__name__,
            'dropout': float(model.stem.dropout.p) if model.stem.dropout is not None else None,
        },
        'body': {
            'class': type(model.body[0]).__name__,
            'kernel_size': int(model.body[0].conv1d.kernel_size[0]),
            'activation': type(model.body[0].act).__name__,
            'dropout': float(model.body[0].dropout.p) if model.body[0].dropout is not None else None,
            'residual': bool(model.body[0].residual),
            'dilations': [int(block.conv1d.dilation[0]) for block in model.body],
        },
        'mix_coeff_units': int(mix.dense1.weight.shape[0]) if hasattr(mix, 'dense1') else None,
        # The MLPs spell it 'act', PooledLogitMixCoeff spells it 'activation'. Both are the
        # output nonlinearity, and neither is a parameter, so both must be recorded here.
        'mix_coeff_activation': _mix_coeff_activation_name(mix),
        'head_class': type(head).__name__,
        'mix_coeff_class': type(mix).__name__,
        'mix_coeff_range': getattr(type(mix), 'mix_coeff_range', None),
        'head_layer_class': type(head.head_target).__name__,
        'use_maximum_target_control_logprob': bool(getattr(head, 'use_maximum_target_control_logprob', False)),
        'control_nograd': bool(getattr(head, 'control_nograd', False)),
        'penalty_class': type(penalty).__name__ if penalty is not None else None,
        'penalty_factor': float(penalty.factor) if penalty is not None else None,
        'anchor_strength': float(getattr(penalty, 'anchor_strength', 0.0)) if penalty is not None else None,
        'n_parameters': sum(p.numel() for p in model.parameters()),
        'n_tasks': int(conv.weight.shape[0]),
    }
    return arch


def save_pretrained(
    model: nn.Module,
    stem: str | pathlib.Path,
    *,
    name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write ``<stem>.safetensors`` and ``<stem>.config.json``.

    The model is described from itself (see :func:`describe`). *extra* is merged into the
    config at top level for provenance blocks such as ``export`` or ``reference``.
    """
    from safetensors.torch import save_file

    stem = pathlib.Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    state = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
    weights_path = stem.with_name(stem.name + WEIGHTS_SUFFIX)
    config_path = stem.with_name(stem.name + CONFIG_SUFFIX)

    arch = describe(model)
    if arch['use_projection'] != state_dict_has_projection(state):
        raise ConfigWeightMismatch(
            'Refusing to write a config that disagrees with the weights: describe() says '
            f'use_projection={arch["use_projection"]} but the state dict '
            f'{"has" if state_dict_has_projection(state) else "has no"} projection weights.'
        )

    payload: dict[str, Any] = {
        'name': name or stem.name,
        'architecture': arch,
        'export': {
            'written_on': datetime.date.today().isoformat(),
            'format': 'safetensors state_dict; architecture in this file',
            'exported_in_eval_mode': not model.training,
        },
    }
    for key, value in (extra or {}).items():
        if key in payload and isinstance(payload[key], dict) and isinstance(value, dict):
            payload[key].update(value)
        else:
            payload[key] = value

    save_file(state, str(weights_path))
    config_path.write_text(json.dumps(payload, indent=1) + '\n')
    return weights_path, config_path


def build_from_config(arch: dict[str, Any]):
    """Construct an (untrained) model matching an ``architecture`` block."""
    from parnet.models import RBPNet

    head_cls = _resolve(arch['head_class'])
    mix_cls = _resolve(arch['mix_coeff_class'])
    head_layer_cls = _resolve(arch['head_layer_class'])
    penalty_cls = _resolve(arch.get('penalty_class'))

    penalty_layer = None
    if penalty_cls is not None:
        penalty_kwargs: dict[str, Any] = {'factor': arch.get('penalty_factor') or 0.0}
        if arch.get('anchor_strength') is not None and penalty_cls is _layers.MixCoeffPenalty:
            penalty_kwargs['anchor_strength'] = arch['anchor_strength']
        penalty_layer = functools.partial(penalty_cls, **penalty_kwargs)

    # PooledProbMixCoeff / PooledProbMixCoeffV2 carry their own width and activation, neither of which
    # is a parameter of the saved state dict.
    mix_kwargs: dict[str, Any] = {}
    if arch.get('mix_coeff_units') is not None:
        mix_kwargs['units'] = arch['mix_coeff_units']
    if arch.get('mix_coeff_activation') is not None:
        keyword = 'activation' if mix_cls is _layers.PooledLogitMixCoeff else 'act'
        mix_kwargs[keyword] = _resolve_activation(arch['mix_coeff_activation'])
    mix_layer = functools.partial(mix_cls, **mix_kwargs) if mix_kwargs else mix_cls

    head_kwargs: dict[str, Any] = {
        'head_layer': head_layer_cls,
        'mix_coeff_layer': mix_layer,
        'penalty_layer': penalty_layer,
        'control_nograd': arch.get('control_nograd', False),
    }
    if head_cls is _layers.TargetControlMix:
        head_kwargs['use_maximum_target_control_logprob'] = arch.get('use_maximum_target_control_logprob', False)

    # The stem's filter count sets the width of the whole residual stream, so it has to be
    # rebuilt at the trained width. Configs written before body_width existed fall back to
    # embedding_dim, which equals the body width exactly when there is no projection.
    body_width = arch.get('body_width')
    if body_width is None:
        if arch['use_projection']:
            raise KeyError(
                "config has no 'body_width' and use_projection is true, so the body width "
                'cannot be inferred from embedding_dim. Re-export this model.'
            )
        body_width = arch.get('embedding_dim') or 128

    stem_spec = arch.get('stem') or {}
    body_spec = arch.get('body') or {}

    stem_kwargs: dict[str, Any] = {'filters': body_width}
    if 'kernel_size' in stem_spec:
        stem_kwargs['kernel_size'] = stem_spec['kernel_size']
    if 'activation' in stem_spec:
        stem_kwargs['activation'] = _resolve_activation(stem_spec['activation'])
    if 'dropout' in stem_spec:
        stem_kwargs['dropout'] = stem_spec['dropout']

    body_kwargs: dict[str, Any] = {'filters': body_width}
    for key in ('kernel_size', 'dropout', 'residual'):
        if key in body_spec:
            body_kwargs[key] = body_spec[key]
    if 'activation' in body_spec:
        body_kwargs['activation'] = _resolve_activation(body_spec['activation'])

    # RBPNet computes each block's dilation as int(dilation ** i). When the config records
    # the trained dilations, honour them exactly instead, so a model trained with a
    # different schedule is rebuilt correctly rather than approximately.
    dilations = body_spec.get('dilations')
    if dilations:
        schedule = list(dilations)

        def body_layer(dilation=None, _schedule=schedule, _calls=[]):  # noqa: B006
            index = len(_calls)
            _calls.append(index)
            return _layers.ResConvBlock1D(dilation=_schedule[index], **body_kwargs)
    else:
        body_layer = functools.partial(_layers.ResConvBlock1D, **body_kwargs)

    model = RBPNet(
        num_tasks=arch['n_tasks'],
        layers=arch['n_body_blocks'],
        # Both the stem and every residual block must be built at the trained width: the
        # block adds its conv output to its own input, so a mismatch is a shape error in
        # the residual add rather than a load error.
        stem_layer=functools.partial(_layers.StemConv1D, **stem_kwargs),
        body_layer=body_layer,
        head_layer=functools.partial(head_cls, **head_kwargs),
        embedding_dim=arch.get('embedding_dim') or 128,
        use_projection=arch['use_projection'],
    )

    if dilations:
        built = [int(block.conv1d.dilation[0]) for block in model.body]
        if built != list(dilations):
            raise ConfigWeightMismatch(f'config records dilations {list(dilations)} but the rebuilt body has {built}')
    return model


def from_pretrained(stem: str | pathlib.Path, *, map_location='cpu', strict: bool = True):
    """Load ``<stem>.safetensors`` + ``<stem>.config.json`` and return the model in eval mode.

    Raises :class:`ConfigWeightMismatch` when the config's ``use_projection`` disagrees
    with the presence of projection weights, rather than trusting the config.
    """
    from safetensors.torch import load_file

    stem = pathlib.Path(stem)
    weights_path = stem.with_name(stem.name + WEIGHTS_SUFFIX)
    config_path = stem.with_name(stem.name + CONFIG_SUFFIX)
    for path in (weights_path, config_path):
        if not path.exists():
            raise FileNotFoundError(f'{path} not found. from_pretrained() takes a stem, not a filename.')

    raw = json.loads(config_path.read_text())
    # Validate before building anything. The architecture block is checked strictly, so a
    # misspelled key fails here instead of being dropped and leaving the builder to fall
    # back on a default that is not what was trained. That failure mode is not
    # hypothetical: configs that omitted the activations loaded cleanly with strict=True
    # and computed something else.
    try:
        parsed = ModelConfig.model_validate(raw)
    except ValidationError as error:
        raise ConfigWeightMismatch(f'{config_path} is not a valid parnet config.\n{error}') from error
    config = raw
    arch = parsed.architecture.model_dump(by_alias=True)
    state = load_file(str(weights_path), device=map_location)

    weights_say = state_dict_has_projection(state)
    if arch.get('use_projection') != weights_say:
        raise ConfigWeightMismatch(
            f'{config_path.name} says use_projection={arch.get("use_projection")}, but the '
            f'weights {"contain" if weights_say else "contain no"} projection parameters. '
            'The weights are authoritative; fix the config.'
        )

    model = build_from_config(arch)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if strict and (missing or unexpected):  # pragma: no cover - load_state_dict raises first
        raise RuntimeError(f'missing={missing} unexpected={unexpected}')
    model.eval()
    model.parnet_config = config
    return model
