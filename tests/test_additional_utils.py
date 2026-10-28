"""Tests for the HPD segmentation helpers."""

import numpy as np
import pytest

from parnet.additional_utils import (
    make_hpd_masks_from_probability_vector,
    make_hpd_segments,
)


def test_the_mask_is_the_smallest_set_holding_the_mass():
    p = np.array([0.5, 0.3, 0.1, 0.06, 0.04])
    mask = make_hpd_masks_from_probability_vector(p, alpha=0.1)
    assert mask.tolist() == [True, True, True, False, False]
    assert p[mask].sum() >= 0.9


def test_a_uniform_distribution_keeps_almost_everything():
    p = np.full(100, 0.01)
    mask = make_hpd_masks_from_probability_vector(p, alpha=0.05)
    # Not pinned exactly: the cumulative sum of 0.01 lands on 0.95 only to within floating
    # point, so which side of the threshold position 95 falls on is not worth asserting.
    assert 94 <= mask.sum() <= 96
    assert p[mask].sum() >= 0.95


def test_alpha_zero_keeps_every_position():
    p = np.array([0.7, 0.2, 0.1])
    assert make_hpd_masks_from_probability_vector(p, alpha=0.0).all()


def test_an_unnormalised_profile_is_refused():
    """A profile that does not sum to 1 is usually a log-space tensor passed by mistake."""
    with pytest.raises(ValueError, match='sum to 1'):
        make_hpd_masks_from_probability_vector(np.array([2.0, 3.0, 5.0]))


def test_a_two_dimensional_input_is_refused():
    with pytest.raises(ValueError, match='1-D'):
        make_hpd_masks_from_probability_vector(np.full((2, 5), 0.1))


def test_the_hpd_set_need_not_be_contiguous():
    """Positions are ranked independently, so an isolated high position joins on merit."""
    p = np.array([0.45, 0.01, 0.45, 0.09])
    segments = make_hpd_segments(p, alpha=0.05)
    assert segments == [(0, 1), (2, 4)]


def test_segments_are_half_open():
    p = np.array([0.0, 0.5, 0.5, 0.0])
    assert make_hpd_segments(p, alpha=0.05) == [(1, 3)]


def test_a_single_block_gives_one_segment():
    p = np.zeros(10)
    p[3:7] = 0.25
    assert make_hpd_segments(p, alpha=0.05) == [(3, 7)]


def test_segments_cover_exactly_the_mask():
    rng = np.random.default_rng(0)
    p = rng.random(200)
    p /= p.sum()
    mask = make_hpd_masks_from_probability_vector(p, alpha=0.2)
    covered = np.zeros_like(mask)
    for start, end in make_hpd_segments(p, alpha=0.2):
        covered[start:end] = True
    assert np.array_equal(covered, mask)


def test_it_works_on_a_softmaxed_model_profile():
    """The documented route from a PARNET output: normalise over the length axis."""
    import torch

    logits = torch.randn(600)
    p = torch.softmax(logits, dim=-1).numpy().astype(np.float64)
    p /= p.sum()
    mask = make_hpd_masks_from_probability_vector(p, alpha=0.05)
    assert 0 < mask.sum() < 600
    assert p[mask].sum() >= 0.95
