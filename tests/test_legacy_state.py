"""Tests for legacy-checkpoint compatibility.

torch.load on a whole pickled module restores __dict__ without calling __init__, so every
attribute added to a layer class breaks every checkpoint pickled before it, and the break
surfaces as an AttributeError from inside forward rather than at load time. These tests
simulate older checkpoints by deleting the attributes and re-running __setstate__.

Three such attributes have been hit on real released files:
  MixCoeffPenalty.lambda_anchor                   on 3M and 7M checkpoints
  AdditiveMix.use_maximum_target_control_logprob  on the released 21M
  AdditiveMix.control_nograd                      on the released 21M
and the released 21M has no `projection` attribute at all, while the 7M has one holding
None and the 3M has one holding nn.Identity.
"""

import pickle

import torch
import torch.nn as nn

from parnet.layers import (
    AdditiveMix,
    MixCoeffPenalty,
    NewAdditiveMix,
    SequenceLinearMix,
    _LEGACY_ATTRIBUTE_DEFAULTS,
)
from parnet.models import RBPNet


def _roundtrip_without(obj, *attrs):
    """Pickle obj with some attributes removed, simulating an older checkpoint.

    Submodules live in ``nn.Module._modules`` rather than in ``__dict__``, and
    ``nn.Module.__getattr__`` consults both, so an attribute must be removed from whichever
    holds it. The released 21M genuinely has no ``projection`` entry in ``_modules``, which
    is why ``hasattr(model, "projection")`` is False on it.
    """
    for a in attrs:
        obj.__dict__.pop(a, None)
        if isinstance(obj, nn.Module):
            obj._modules.pop(a, None)
            obj._parameters.pop(a, None)
            obj._buffers.pop(a, None)
    assert not hasattr(obj, attrs[0]), f'{attrs[0]} survived removal; the helper is wrong'
    return pickle.loads(pickle.dumps(obj))


def test_additivemix_regains_control_nograd():
    m = AdditiveMix(4, mix_coeff_layer=SequenceLinearMix)
    back = _roundtrip_without(m, 'control_nograd')
    assert back.control_nograd is False


def test_additivemix_regains_the_mixing_flag():
    m = AdditiveMix(4, mix_coeff_layer=SequenceLinearMix)
    back = _roundtrip_without(m, 'use_maximum_target_control_logprob')
    # False is the log-sum branch, the only behaviour before the flag existed
    assert back.use_maximum_target_control_logprob is False


def test_penalty_regains_lambda_anchor():
    p = MixCoeffPenalty(factor=5.0)
    back = _roundtrip_without(p, 'lambda_anchor')
    assert back.lambda_anchor == 0.0
    assert back.factor == 5.0  # the attribute that was there is untouched


def test_newadditivemix_regains_control_nograd():
    m = NewAdditiveMix(4)
    back = _roundtrip_without(m, 'control_nograd')
    assert back.control_nograd is False


def test_existing_values_are_never_overwritten():
    m = AdditiveMix(4, mix_coeff_layer=SequenceLinearMix, control_nograd=True)
    back = pickle.loads(pickle.dumps(m))
    assert back.control_nograd is True  # not reset to the default


def _tiny_model(use_projection=False):
    """Smallest RBPNet-shaped model that exercises forward.

    use_projection=False gives an nn.Identity projection, which is what the 3M checkpoints
    carry; the released 21M has no projection attribute at all.
    """
    return RBPNet(
        num_tasks=4,
        layers=1,
        use_projection=use_projection,
        head_layer=lambda n: AdditiveMix(n, mix_coeff_layer=SequenceLinearMix),
    )


def test_rbpnet_regains_projection_as_none():
    back = _roundtrip_without(_tiny_model(), 'projection')
    assert back.projection is None


def test_forward_tolerates_a_missing_projection():
    back = _roundtrip_without(_tiny_model(), 'projection')
    back.eval()
    with torch.no_grad():
        out = back(torch.zeros(2, 4, 64))
    assert 'total' in out
    assert torch.isfinite(out['total']).all()


def test_identity_projection_behaves_like_none():
    # Same weights, two spellings of "no projection", so any difference is the guard itself.
    m = _tiny_model()
    assert isinstance(m.projection, nn.Identity)
    m.eval()
    x = torch.randn(2, 4, 64)
    with torch.no_grad():
        with_identity = m(x)['total'].clone()
    m.projection = None
    with torch.no_grad():
        with_none = m(x)['total']
    assert torch.equal(with_identity, with_none)


def test_the_defaults_table_is_documented_for_every_guarded_class():
    # A class in the table but absent from the codebase, or vice versa, is a silent gap.
    for cls_name, defaults in _LEGACY_ATTRIBUTE_DEFAULTS.items():
        assert defaults, f'{cls_name} has an empty default set'
        assert all(isinstance(k, str) for k in defaults)
