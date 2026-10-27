"""Tests for parnet.finetuning.

The point of the module is that "unfreeze the last N" means the same thing whether or not
the model has a trainable projection, so most of these tests are the same assertion made
against three projection variants.
"""

import pytest
import torch
import torch.nn as nn

from parnet.finetuning import (
    describe_trainable,
    set_trainable,
    stage_names,
    trainable_stages,
)


class _Block(nn.Module):
    def __init__(self, c=8):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 3, padding="same")

    def forward(self, x):
        return self.conv(x)


class _Fake(nn.Module):
    """RBPNet-shaped stand-in: stem, body of 3 blocks, a projection, a head."""

    def __init__(self, projection):
        super().__init__()
        self.stem = nn.Conv1d(4, 8, 3, padding="same")
        self.body = nn.Sequential(_Block(), _Block(), _Block())
        self.projection = projection
        self.head = nn.Conv1d(8, 2, 1)


def _no_projection():
    return _Fake(None)


def _identity_projection():
    return _Fake(nn.Identity())


def _real_projection():
    return _Fake(nn.Conv1d(8, 8, 1, bias=False))


@pytest.mark.parametrize("make", [_no_projection, _identity_projection])
def test_parameterless_projection_is_not_a_stage(make):
    # This is the whole point: None and Identity must behave identically, so a notebook
    # written for one works on the other.
    assert stage_names(make()) == ["stem", "body.0", "body.1", "body.2"]


def test_real_projection_is_a_stage():
    assert stage_names(_real_projection()) == ["stem", "body.0", "body.1", "body.2", "projection"]


def test_none_and_identity_give_identical_stage_lists():
    assert stage_names(_no_projection()) == stage_names(_identity_projection())


@pytest.mark.parametrize("make", [_no_projection, _identity_projection, _real_projection])
def test_unfreeze_zero_freezes_the_whole_backbone(make):
    m = make()
    rep = set_trainable(m, unfreeze_last=0)
    assert rep["unfrozen"] == []
    for _, mod in trainable_stages(m):
        assert all(not p.requires_grad for p in mod.parameters())
    assert all(p.requires_grad for p in m.head.parameters())


@pytest.mark.parametrize("make", [_no_projection, _identity_projection, _real_projection])
def test_unfreeze_minus_one_unfreezes_everything(make):
    m = make()
    rep = set_trainable(m, unfreeze_last=-1)
    assert rep["unfrozen"] == rep["stages"]
    assert rep["n_frozen"] == 0


def test_unfreeze_last_counts_from_the_output_side():
    m = _real_projection()
    rep = set_trainable(m, unfreeze_last=2)
    # with a real projection the last two stages are body.2 and projection
    assert rep["unfrozen"] == ["body.2", "projection"]
    assert all(p.requires_grad for p in m.projection.parameters())
    assert not any(p.requires_grad for p in m.body[0].parameters())


def test_same_request_on_a_model_without_a_projection():
    m = _identity_projection()
    rep = set_trainable(m, unfreeze_last=2)
    # no projection stage, so the last two are body.1 and body.2, not body.2 and a no-op
    assert rep["unfrozen"] == ["body.1", "body.2"]


def test_too_many_stages_is_an_error_not_a_clamp():
    m = _identity_projection()
    with pytest.raises(ValueError, match="4 trainable backbone stages"):
        set_trainable(m, unfreeze_last=99)


def test_negative_other_than_minus_one_is_rejected():
    with pytest.raises(ValueError, match="must be -1, 0 or positive"):
        set_trainable(_identity_projection(), unfreeze_last=-2)


def test_head_can_be_frozen_too():
    m = _identity_projection()
    rep = set_trainable(m, unfreeze_last=0, head=False)
    assert rep["head_trainable"] is False
    assert rep["n_trainable"] == 0


def test_counts_add_up():
    m = _real_projection()
    set_trainable(m, unfreeze_last=1)
    d = describe_trainable(m)
    assert d["n_trainable"] + d["n_frozen"] == d["n_total"]
    assert d["n_trainable"] > 0


def test_report_is_printable_and_complete():
    rep = set_trainable(_real_projection(), unfreeze_last=1)
    for key in ("stages", "unfrozen", "head_trainable", "n_trainable", "n_frozen", "n_total"):
        assert key in rep
