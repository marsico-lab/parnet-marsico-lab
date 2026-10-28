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

import torch
import torch.nn as nn

from parnet import layers as _layers

#: Classes that may be named in a ``config.json``. Resolution goes through this table and
#: never through ``getattr`` on a module, so a config file cannot cause an arbitrary
#: import. Adding a class here is a deliberate act.
CLASS_REGISTRY: dict[str, type] = {
    'StemConv1D': _layers.StemConv1D,
    'ResConvBlock1D': _layers.ResConvBlock1D,
    'LinearProjectionHead': _layers.LinearProjectionHead,
    'LinearProjection': _layers.LinearProjection,
    'SequenceLinearMix': _layers.SequenceLinearMix,
    'MixCoeffMLP': _layers.MixCoeffMLP,
    'NewMixCoeffMLP': _layers.NewMixCoeffMLP,
    'MixCoeffPenalty': _layers.MixCoeffPenalty,
    'NewMixCoeffPenalty': _layers.NewMixCoeffPenalty,
    'AdditiveMix': _layers.AdditiveMix,
    'NewAdditiveMix': _layers.NewAdditiveMix,
}

CONFIG_SUFFIX = '.config.json'
WEIGHTS_SUFFIX = '.safetensors'


class ConfigWeightMismatch(ValueError):
    """Raised when ``config.json`` claims something the weights contradict."""


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
    return any(
        key.startswith('projection.') and key.endswith(('.weight', '.bias'))
        for key in state_dict
    )


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
        'embedding_dim': int(conv.weight.shape[1]),
        'use_projection': has_projection,
        'head_class': type(head).__name__,
        'mix_coeff_class': type(mix).__name__,
        'mix_coeff_range': getattr(type(mix), 'mix_coeff_range', None),
        'head_layer_class': type(head.head_target).__name__,
        'use_maximum_target_control_logprob': bool(
            getattr(head, 'use_maximum_target_control_logprob', False)
        ),
        'control_nograd': bool(getattr(head, 'control_nograd', False)),
        'penalty_class': type(penalty).__name__ if penalty is not None else None,
        'penalty_factor': float(penalty.factor) if penalty is not None else None,
        'lambda_anchor': float(getattr(penalty, 'lambda_anchor', 0.0)) if penalty is not None else None,
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
            f"use_projection={arch['use_projection']} but the state dict "
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
        if arch.get('lambda_anchor') is not None and penalty_cls is _layers.MixCoeffPenalty:
            penalty_kwargs['lambda_anchor'] = arch['lambda_anchor']
        penalty_layer = functools.partial(penalty_cls, **penalty_kwargs)

    head_kwargs: dict[str, Any] = {
        'head_layer': head_layer_cls,
        'mix_coeff_layer': mix_cls,
        'penalty_layer': penalty_layer,
        'control_nograd': arch.get('control_nograd', False),
    }
    if head_cls is _layers.AdditiveMix:
        head_kwargs['use_maximum_target_control_logprob'] = arch.get(
            'use_maximum_target_control_logprob', False
        )

    return RBPNet(
        num_tasks=arch['n_tasks'],
        layers=arch['n_body_blocks'],
        head_layer=functools.partial(head_cls, **head_kwargs),
        embedding_dim=arch.get('embedding_dim') or 128,
        use_projection=arch['use_projection'],
    )


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

    config = json.loads(config_path.read_text())
    arch = dict(config['architecture'])
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
