"""Padding, kept separate from tokenization.

Tokenizing and padding are different decisions. Tokenizing is a lookup with one right
answer; padding is a modelling choice about where the real sequence sits inside the
window, and a sequence-to-signal model is position-sensitive, so getting it wrong shifts
every prediction. Folding padding into the tokenizer hides that choice behind a boolean.

The centre convention here is **ceil-left, floor-right**, matching the pre-padded 600 nt
source data and ``parnet_additional_utils.patch_datasets``. Note that
``parnet.data.datasets.ListDataset`` uses the opposite tie-break (floor-left, ceil-right);
for odd total padding the two disagree by one nucleotide. This module follows the data.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from parnet.tokenization import PAD_INDEX, TokenizationError

#: ``meta['pad_side']`` codes used by the dataset layer, mapped to the names used here.
#: ``-1`` and ``None`` both mean "do not pad".
PAD_SIDE_CODES: dict[int, str | None] = {0: 'center', 1: 'left', 2: 'right', -1: None}

_SIDES = frozenset({'left', 'right', 'center'})


def _normalise_side(side: str | int | None) -> str | None:
    if isinstance(side, bool):  # bool is an int subclass; reject it explicitly
        raise TokenizationError(f'side must be a name or a pad_side code, got {side!r}')
    if isinstance(side, int):
        if side not in PAD_SIDE_CODES:
            raise TokenizationError(f'unknown pad_side code {side!r}; known codes are {sorted(PAD_SIDE_CODES)}')
        return PAD_SIDE_CODES[side]
    if side is None or side in _SIDES:
        return side
    raise TokenizationError(f'side must be one of {sorted(_SIDES)}, a pad_side code, or None; got {side!r}')


def pad_sizes(length: int, target: int, side: str | int | None) -> tuple[int, int]:
    """Return ``(left, right)`` padding for one sequence.

    ``center`` puts the extra nucleotide on the **left** when the total padding is odd,
    which is the convention the 600 nt source tiles were built with.
    """
    name = _normalise_side(side)
    deficit = target - length
    if deficit < 0:
        raise TokenizationError(
            f'sequence of length {length} is longer than the target {target}; '
            'padding does not truncate, so shorten the sequence yourself if that is what you want'
        )
    if name is None or deficit == 0:
        return 0, 0
    if name == 'left':
        return deficit, 0
    if name == 'right':
        return 0, deficit
    return deficit - deficit // 2, deficit // 2  # center: ceil-left, floor-right


def pad_sequences(
    encoded: Sequence[Sequence[int]] | Sequence[torch.Tensor] | dict,
    *,
    side: str | int | None | Sequence[str | int | None] = 'right',
    length: int | None = None,
    pad_index: int = PAD_INDEX,
) -> dict[str, torch.Tensor]:
    """Pad encoded sequences to a common length.

    Args:
        encoded: What ``RnaTokenizer(..., batched=False)`` returns, or the plain list of
            per-sequence id lists or 1-D tensors inside it.
        side: ``"left"``, ``"right"``, ``"center"``, a ``pad_side`` code from
            :data:`PAD_SIDE_CODES`, or ``None`` for no padding. Pass a sequence of these
            to give each input its own side.
        length: Target length. Defaults to the longest sequence present.
        pad_index: Index written into padded positions. The default is ``N``, an all-zero
            one-hot column, so padding contributes nothing to any convolution.

    Returns:
        ``{"input_ids": (B, L) long, "attention_mask": (B, L) long}``. The mask is 1 at
        real positions and 0 at padding, always returned so callers never have to infer
        which positions were invented.
    """
    if isinstance(encoded, dict):
        encoded = encoded['input_ids']
    items = [list(x.tolist()) if isinstance(x, torch.Tensor) else list(x) for x in encoded]
    if len(items) == 0:
        raise TokenizationError('no sequences given')

    lengths = [len(x) for x in items]
    target = max(lengths) if length is None else length

    if isinstance(side, (str, int)) or side is None:
        sides: list = [side] * len(items)
    else:
        sides = list(side)
        if len(sides) != len(items):
            raise TokenizationError(
                f'got {len(sides)} sides for {len(items)} sequences; pass one side or one per sequence'
            )

    ids, mask = [], []
    for item, own_length, own_side in zip(items, lengths, sides):
        left, right = pad_sizes(own_length, target, own_side)
        if left + right + own_length != target:
            raise TokenizationError(
                f'sequence of length {own_length} cannot reach {target} with side={own_side!r}; '
                'side=None means no padding, so it cannot be combined with a longer target'
            )
        ids.append([pad_index] * left + item + [pad_index] * right)
        mask.append([0] * left + [1] * own_length + [0] * right)

    return {
        'input_ids': torch.tensor(ids, dtype=torch.long),
        'attention_mask': torch.tensor(mask, dtype=torch.long),
    }
