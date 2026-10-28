"""Helpers that sit next to the model rather than inside it.

The segmentation functions here turn a predicted profile into intervals. They are copied
from ``parnet_analyses_libs.span_masks``, where they are maintained, so that the tutorials
and anything else built on PARNET can use them without taking on that package's dependency
tree, which pins TensorFlow, pysam, pyBigWig, logomaker and pandera for what is twenty
lines of NumPy. The docstrings were converted to Google style on the way in.

Highest-posterior-density segmentation is **one** way to decide which positions of a
predicted profile count as bound. It is not an official PARNET peak caller and there is no
such thing here; ``scipy.signal.find_peaks`` on the same profile is an equally defensible
choice, and an IDR narrowPeak set is a different kind of object again. What HPD gives you
is a statement about probability mass rather than about peak shape: the smallest set of
positions holding ``1 - alpha`` of the predicted signal.

Two sharp edges, both inherited from the definition rather than the implementation:

- The input must be **normalised**. A PARNET profile is a shape, not a rate, so use
  ``to_probs=True`` or an explicit softmax over the length axis. The mask function raises
  rather than renormalising silently, because a profile that does not sum to one usually
  means a log-space tensor was passed by mistake.
- An HPD set is **not necessarily contiguous**. Positions are ranked independently, so a
  high-scoring isolated nucleotide joins the set on its own merits. Call sites that want
  interval-like output usually drop runs of length one.
"""

from __future__ import annotations

import numpy as np

__all__ = ['make_hpd_masks_from_probability_vector', 'make_hpd_segments']


def _coordinates_from_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Return the half-open coordinates of each run of ones in a binary mask.

    Args:
        binary_mask: 1-D integer array of shape ``(length,)``.

    Returns:
        Int64 array of shape ``(n_regions, 2)``; each row is ``[start, end)``.
    """
    return np.where(np.diff(binary_mask, prepend=0, append=0) != 0)[0].reshape(-1, 2).astype(np.int64)


def make_hpd_masks_from_probability_vector(
    p: np.ndarray,
    alpha: float = 0.05,
    atol: float = 1e-8,
) -> np.ndarray:
    """Build the highest-posterior-density mask of a discrete distribution.

    Positions are taken in decreasing order of probability until ``1 - alpha`` of the mass
    is covered, so the result is the smallest set of positions holding that much of the
    predicted signal.

    Args:
        p: 1-D array of probabilities summing to 1.
        alpha: Mass allowed to fall outside the set, so ``0.05`` gives the 95 % HPD.
        atol: Absolute tolerance used when checking that ``p`` sums to 1.

    Returns:
        Boolean array of the same shape as ``p``, True at HPD positions. The set is not
        guaranteed to be contiguous.

    Raises:
        ValueError: If ``p`` is not 1-D, or does not sum to 1 within ``atol``. The second
            case is usually a log-space profile passed by mistake; normalise it first
            rather than relaxing the tolerance.
    """
    if p.ndim != 1:
        raise ValueError('p must be 1-D')

    if not np.isclose(p.sum(), 1.0, atol=atol):
        raise ValueError('p must sum to 1')

    idx_sorted = np.argsort(p)[::-1]
    cumsum = np.cumsum(p[idx_sorted])
    k = np.searchsorted(cumsum, 1 - alpha) + 1
    hpd_idx = idx_sorted[:k]

    mask = np.zeros_like(p, dtype=bool)
    mask[hpd_idx] = True
    return mask


def make_hpd_segments(
    p: np.ndarray,
    alpha: float = 0.05,
    atol: float = 1e-8,
) -> list[tuple[int, int]]:
    """Segment the highest-posterior-density set into intervals.

    Args:
        p: 1-D array of probabilities summing to 1.
        alpha: Mass allowed to fall outside the set, so ``0.05`` gives the 95 % HPD.
        atol: Absolute tolerance used when checking that ``p`` sums to 1.

    Returns:
        List of half-open ``(start, end)`` intervals covering the HPD positions. Because
        the HPD set is ranked position-wise, isolated nucleotides appear as length-1
        intervals; filter them if you want peak-like spans.

    Raises:
        ValueError: If ``p`` is not 1-D, or does not sum to 1 within ``atol``.
    """
    mask = make_hpd_masks_from_probability_vector(p, alpha=alpha, atol=atol)
    coords = _coordinates_from_binary_mask(mask.astype(np.int64))
    return [(int(start), int(end)) for start, end in coords]
