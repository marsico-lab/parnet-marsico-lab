"""Tests for the safetensors + config.json model format."""

import functools
import json

import pytest
import torch

from parnet import layers
from parnet.models import RBPNet
from parnet.pretrained import (
    ConfigWeightMismatch,
    describe,
    from_pretrained,
    save_pretrained,
    state_dict_has_projection,
)


def _small_model(use_projection=False, num_tasks=3, bounded=True):
    """A deliberately tiny RBPNet so the suite stays fast.

    The head is specified explicitly rather than left to TargetControlMix's defaults, because
    those defaults are themselves an incompatible pair: PooledProbMixCoeff emits a bounded
    coefficient while use_maximum_target_control_logprob=False selects the log-sum rule,
    which needs a logit. Constructing one now raises IncompatibleMixError, which is the
    intended behaviour and is covered by tests/test_mix_guard.py.
    """
    if bounded:
        # bounded coefficient + convex combination (the 7M and 3M pairing)
        head = functools.partial(
            layers.TargetControlMix,
            mix_coeff_layer=layers.PooledProbMixCoeff,
            use_maximum_target_control_logprob=True,
        )
    else:
        # unbounded logit + log-space addition (the released 21M pairing)
        head = functools.partial(
            layers.TargetControlMix,
            mix_coeff_layer=layers.PooledLogitMixCoeff,
            use_maximum_target_control_logprob=False,
        )
    return RBPNet(
        num_tasks=num_tasks,
        layers=2,
        embedding_dim=16,
        use_projection=use_projection,
        head_layer=head,
    )


def test_round_trip_reproduces_outputs_bitwise(tmp_path):
    torch.manual_seed(0)
    model = _small_model()
    model.eval()
    x = torch.rand(2, 4, 64)
    with torch.no_grad():
        before = model(x)

    save_pretrained(model, tmp_path / 'tiny')
    reloaded = from_pretrained(tmp_path / 'tiny')
    with torch.no_grad():
        after = reloaded(x)

    for key in before:
        assert torch.equal(before[key], after[key]), f'{key} changed across a save/load round trip'


def test_loaded_model_is_in_eval_mode(tmp_path):
    """A caller who forgets .eval() would otherwise get batch statistics and live dropout."""
    model = _small_model()
    model.train()
    save_pretrained(model, tmp_path / 'tiny')
    reloaded = from_pretrained(tmp_path / 'tiny')
    assert reloaded.training is False
    assert all(not m.training for m in reloaded.modules())


def test_eval_mode_is_deterministic(tmp_path):
    model = _small_model()
    save_pretrained(model, tmp_path / 'tiny')
    reloaded = from_pretrained(tmp_path / 'tiny')
    x = torch.rand(2, 4, 64)
    with torch.no_grad():
        assert torch.equal(reloaded(x)['total'], reloaded(x)['total'])


def test_no_projection_is_reported_as_false(tmp_path):
    model = _small_model(use_projection=False)
    save_pretrained(model, tmp_path / 'tiny')
    config = json.loads((tmp_path / 'tiny.config.json').read_text())
    assert config['architecture']['use_projection'] is False
    assert not state_dict_has_projection(model.state_dict())


def test_real_projection_is_reported_as_true(tmp_path):
    model = _small_model(use_projection=True)
    save_pretrained(model, tmp_path / 'tiny')
    config = json.loads((tmp_path / 'tiny.config.json').read_text())
    assert config['architecture']['use_projection'] is True
    assert state_dict_has_projection(model.state_dict())
    assert from_pretrained(tmp_path / 'tiny') is not None


def test_config_disagreeing_with_weights_is_rejected(tmp_path):
    """The whole point of the format: weights win, and a lying config is an error.

    This is not hypothetical. The first export of the 3M backbones wrote
    use_projection=true for models whose projection was a parameterless nn.Identity,
    because it tested `projection is not None` rather than looking for weights.
    """
    model = _small_model(use_projection=False)
    save_pretrained(model, tmp_path / 'tiny')
    config_path = tmp_path / 'tiny.config.json'
    config = json.loads(config_path.read_text())
    config['architecture']['use_projection'] = True
    config_path.write_text(json.dumps(config))

    with pytest.raises(ConfigWeightMismatch, match='weights are authoritative'):
        from_pretrained(tmp_path / 'tiny')


def test_unknown_class_in_config_is_rejected(tmp_path):
    """A config file must not be able to name an arbitrary importable object."""
    model = _small_model()
    save_pretrained(model, tmp_path / 'tiny')
    config_path = tmp_path / 'tiny.config.json'
    config = json.loads(config_path.read_text())
    config['architecture']['mix_coeff_class'] = 'os.system'
    config_path.write_text(json.dumps(config))

    with pytest.raises(KeyError, match='not a registered PARNET class'):
        from_pretrained(tmp_path / 'tiny')


def test_describe_reads_the_model_not_its_arguments():
    model = _small_model(num_tasks=5)
    arch = describe(model)
    assert arch['n_tasks'] == 5
    assert arch['head_class'] == 'TargetControlMix'
    assert arch['mix_coeff_class'] in {'PooledProbMixCoeff', 'PooledProbMixCoeffV2', 'PooledLogitMixCoeff'}
    assert arch['n_parameters'] == sum(p.numel() for p in model.parameters())


def test_from_pretrained_requires_a_stem_not_a_filename(tmp_path):
    model = _small_model()
    save_pretrained(model, tmp_path / 'tiny')
    with pytest.raises(FileNotFoundError, match='takes a stem'):
        from_pretrained(tmp_path / 'tiny.safetensors')


def test_mix_coeff_range_survives_the_round_trip(tmp_path):
    """mix_coeff_range decides which combination rule is legal; losing it loses the guard."""
    model = _small_model(num_tasks=2, bounded=False)
    save_pretrained(model, tmp_path / 'tiny')
    config = json.loads((tmp_path / 'tiny.config.json').read_text())
    assert config['architecture']['mix_coeff_range'] == 'logit'
    assert config['architecture']['use_maximum_target_control_logprob'] is False
    reloaded = from_pretrained(tmp_path / 'tiny')
    assert type(reloaded.head.mix_coeff).__name__ == 'PooledLogitMixCoeff'


def test_body_width_is_recorded_and_restored(tmp_path):
    """A model wider than the default stem must come back at its trained width.

    Regression test: the first version of build_from_config ignored the stem's filter
    count, so every model was rebuilt at the default 128 channels and a 768-channel
    checkpoint failed to load with a wall of size mismatches.
    """
    model = RBPNet(
        num_tasks=2,
        layers=2,
        embedding_dim=16,
        use_projection=False,
        stem_layer=functools.partial(layers.StemConv1D, filters=48),
        body_layer=functools.partial(layers.ResConvBlock1D, filters=48),
        head_layer=functools.partial(
            layers.TargetControlMix,
            mix_coeff_layer=layers.PooledProbMixCoeff,
            use_maximum_target_control_logprob=True,
        ),
    )
    model.eval()
    x = torch.rand(2, 4, 64)
    with torch.no_grad():
        before = model(x)

    save_pretrained(model, tmp_path / 'wide')
    config = json.loads((tmp_path / 'wide.config.json').read_text())
    assert config['architecture']['body_width'] == 48

    reloaded = from_pretrained(tmp_path / 'wide')
    assert reloaded.stem.conv1d.weight.shape[0] == 48
    assert all(block.conv1d.weight.shape[0] == 48 for block in reloaded.body)
    with torch.no_grad():
        after = reloaded(x)
    for key in before:
        assert torch.equal(before[key], after[key])


def test_non_parametric_architecture_survives_the_round_trip(tmp_path):
    """Activations, kernel sizes and dilations are not in the state dict.

    Regression test for a real defect: the 3M backbones were trained with GELU, the
    config did not record it, and the rebuilt model used the default ReLU. Weights loaded
    cleanly with strict=True and the model computed something different, which only
    surfaced because the export stored reference outputs.
    """
    model = RBPNet(
        num_tasks=2,
        layers=3,
        embedding_dim=16,
        use_projection=False,
        stem_layer=functools.partial(layers.StemConv1D, filters=24, activation=torch.nn.GELU()),
        body_layer=functools.partial(layers.ResConvBlock1D, filters=24, activation=torch.nn.GELU()),
        head_layer=functools.partial(
            layers.TargetControlMix,
            mix_coeff_layer=functools.partial(layers.PooledProbMixCoeff, units=32, act=torch.nn.GELU()),
            use_maximum_target_control_logprob=True,
        ),
    )
    model.eval()
    x = torch.rand(2, 4, 64)
    with torch.no_grad():
        before = model(x)

    save_pretrained(model, tmp_path / 'gelu')
    config = json.loads((tmp_path / 'gelu.config.json').read_text())['architecture']
    assert config['stem']['activation'] == 'GELU'
    assert config['body']['activation'] == 'GELU'
    assert config['mix_coeff_activation'] == 'GELU'
    assert config['mix_coeff_units'] == 32
    assert config['body']['dilations'] == [1, 1, 3]

    reloaded = from_pretrained(tmp_path / 'gelu')
    assert type(reloaded.stem.act).__name__ == 'GELU'
    assert type(reloaded.body[0].act).__name__ == 'GELU'
    assert type(reloaded.head.mix_coeff.act).__name__ == 'GELU'
    with torch.no_grad():
        after = reloaded(x)
    for key in before:
        assert torch.equal(before[key], after[key]), f'{key} differs after round trip'


def test_unknown_activation_is_rejected(tmp_path):
    model = _small_model()
    save_pretrained(model, tmp_path / 'tiny')
    path = tmp_path / 'tiny.config.json'
    config = json.loads(path.read_text())
    config['architecture']['body']['activation'] = 'os.system'
    path.write_text(json.dumps(config))
    with pytest.raises(KeyError, match='not a registered activation'):
        from_pretrained(tmp_path / 'tiny')


def test_embed_works_when_projection_is_none():
    """embed() called self.projection(x) unguarded, unlike forward().

    Legacy checkpoints get projection=None from _LegacyStateMixin, so embed() raised
    TypeError: 'NoneType' object is not callable from inside the library.
    """
    model = RBPNet(
        num_tasks=2,
        layers=2,
        embedding_dim=16,
        use_projection=False,
        head_layer=functools.partial(
            layers.TargetControlMix, mix_coeff_layer=layers.PooledProbMixCoeff, use_maximum_target_control_logprob=True
        ),
    )
    model.eval()
    model.projection = None  # exactly what the legacy loader leaves behind

    x = torch.rand(2, 4, 64)
    with torch.no_grad():
        full = model.embed(x)
        mean = model.embed(x, aggregation='mean')
        center = model.embed(x, aggregation='center')
        assert torch.equal(model(x)['total'], model(x)['total'])

    assert full.shape[0] == 2 and full.shape[-1] == 64
    assert mean.shape == full.shape[:2]
    assert center.shape == full.shape[:2]


def test_embedding_width_matches_what_the_head_receives():
    """embed() must return the width a head would see, which is the body width here."""
    model = RBPNet(
        num_tasks=2,
        layers=2,
        embedding_dim=16,
        use_projection=False,
        stem_layer=functools.partial(layers.StemConv1D, filters=24),
        body_layer=functools.partial(layers.ResConvBlock1D, filters=24),
        head_layer=functools.partial(
            layers.TargetControlMix, mix_coeff_layer=layers.PooledProbMixCoeff, use_maximum_target_control_logprob=True
        ),
    )
    model.eval()
    with torch.no_grad():
        assert model.embed(torch.rand(1, 4, 64)).shape[1] == 24
