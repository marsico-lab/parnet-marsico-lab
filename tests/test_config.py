"""Tests for the config.json schema."""

import json

import pytest
from pydantic import ValidationError

from parnet.config import ArchitectureConfig, ModelConfig


def _architecture(**overrides):
    base = {
        'class': 'RBPNet',
        'n_body_blocks': 2,
        'body_width': 32,
        'embedding_dim': 32,
        'use_projection': False,
        'stem': {'class': 'StemConv1D', 'kernel_size': 12, 'activation': 'GELU', 'dropout': None},
        'body': {
            'class': 'ResConvBlock1D',
            'kernel_size': 3,
            'activation': 'GELU',
            'dropout': 0.25,
            'residual': True,
            'dilations': [1, 1],
        },
        'mix_coeff_units': 128,
        'mix_coeff_activation': 'ReLU',
        'head_class': 'AdditiveMix',
        'mix_coeff_class': 'MixCoeffMLP',
        'mix_coeff_range': 'unit',
        'head_layer_class': 'LinearProjectionHead',
        'use_maximum_target_control_logprob': True,
        'control_nograd': False,
        'penalty_class': 'MixCoeffPenalty',
        'penalty_factor': 5.0,
        'lambda_anchor': 0.0,
        'n_parameters': 1234,
        'n_tasks': 51,
    }
    base.update(overrides)
    return base


def test_a_valid_architecture_parses():
    arch = ArchitectureConfig.model_validate(_architecture())
    assert arch.class_name == 'RBPNet'
    assert arch.body.dilations == [1, 1]
    assert arch.stem.activation == 'GELU'


def test_a_typo_in_the_architecture_is_rejected():
    """The whole point of strict validation: a misspelled key must not be ignored.

    Without extra='forbid', 'activaton' would be dropped, the default ReLU would be used,
    and the model would load cleanly while computing something else.
    """
    bad = _architecture()
    bad['stem'] = dict(bad['stem'])
    bad['stem']['activaton'] = 'GELU'
    with pytest.raises(ValidationError, match='activaton'):
        ArchitectureConfig.model_validate(bad)


def test_an_unknown_top_level_architecture_key_is_rejected():
    with pytest.raises(ValidationError):
        ArchitectureConfig.model_validate(_architecture(n_layerz=9))


def test_a_missing_dilation_schedule_is_rejected():
    bad = _architecture()
    bad['body'] = {k: v for k, v in bad['body'].items() if k != 'dilations'}
    with pytest.raises(ValidationError, match='dilations'):
        ArchitectureConfig.model_validate(bad)


def test_mix_coeff_range_is_constrained():
    with pytest.raises(ValidationError):
        ArchitectureConfig.model_validate(_architecture(mix_coeff_range='probability'))


def test_provenance_blocks_stay_permissive():
    """export and reference grow over time; an unknown key there is a note, not a hazard."""
    config = ModelConfig.model_validate(
        {
            'name': 'tiny',
            'architecture': _architecture(),
            'export': {'written_on': '2026-10-28', 'something_new': 1},
            'reference': {'seed': 20261027, 'mean_target_weight_w': 0.28},
        }
    )
    assert config.export['something_new'] == 1
    assert config.unverifiable_claims == []


def test_unverifiable_claims_round_trip():
    config = ModelConfig.model_validate(
        {
            'name': 'parnet-21m-penalty5',
            'architecture': _architecture(penalty_class=None, penalty_factor=None),
            'unverifiable_claims': [
                {
                    'quantity': 'penalty factor (lambda)',
                    'value': 5.0,
                    'source': 'filename',
                    'recoverable_from_weights': False,
                }
            ],
        }
    )
    claim = config.unverifiable_claims[0]
    assert claim.value == 5.0
    assert claim.recoverable_from_weights is False
    assert config.architecture.penalty_factor is None, 'the file itself carries no penalty'


def test_every_exported_config_validates(tmp_path):
    """Round-trip the schema against a config written by save_pretrained itself."""
    import functools

    import torch

    from parnet import layers
    from parnet.models import RBPNet
    from parnet.pretrained import save_pretrained

    model = RBPNet(
        num_tasks=3,
        layers=2,
        embedding_dim=16,
        use_projection=False,
        head_layer=functools.partial(
            layers.AdditiveMix,
            mix_coeff_layer=layers.MixCoeffMLP,
            use_maximum_target_control_logprob=True,
        ),
    )
    model.eval()
    _, config_path = save_pretrained(model, tmp_path / 'tiny')
    parsed = ModelConfig.model_validate(json.loads(config_path.read_text()))
    assert parsed.architecture.n_tasks == 3
    assert isinstance(parsed.architecture.body.dilations, list)
    del torch
