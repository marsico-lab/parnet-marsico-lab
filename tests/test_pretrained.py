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

    The head is specified explicitly rather than left to AdditiveMix's defaults, because
    those defaults are themselves an incompatible pair: MixCoeffMLP emits a bounded
    coefficient while use_maximum_target_control_logprob=False selects the log-sum rule,
    which needs a logit. Constructing one now raises IncompatibleMixError, which is the
    intended behaviour and is covered by tests/test_mix_guard.py.
    """
    if bounded:
        # bounded coefficient + convex combination (the 7M and 3M pairing)
        head = functools.partial(
            layers.AdditiveMix,
            mix_coeff_layer=layers.MixCoeffMLP,
            use_maximum_target_control_logprob=True,
        )
    else:
        # unbounded logit + log-space addition (the released 21M pairing)
        head = functools.partial(
            layers.AdditiveMix,
            mix_coeff_layer=layers.SequenceLinearMix,
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
    assert arch['head_class'] == 'AdditiveMix'
    assert arch['mix_coeff_class'] in {'MixCoeffMLP', 'NewMixCoeffMLP', 'SequenceLinearMix'}
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
    assert type(reloaded.head.mix_coeff).__name__ == 'SequenceLinearMix'
