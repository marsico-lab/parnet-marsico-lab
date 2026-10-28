"""Tests for padding, kept separate from tokenization."""

import pytest
import torch

from parnet.padding import PAD_SIDE_CODES, pad_sequences, pad_sizes
from parnet.tokenization import PAD_INDEX, RnaTokenizer, TokenizationError


@pytest.fixture
def tok():
    return RnaTokenizer()


# -- the tokenizer no longer pads ----------------------------------------------------


def test_batched_requires_equal_lengths(tok):
    out = tok(['ACGT', 'UGCA'])
    assert out['input_ids'].shape == (2, 4)
    assert 'attention_mask' not in out, 'nothing was invented, so there is nothing to mask'


def test_batched_rejects_ragged_and_names_the_way_out(tok):
    with pytest.raises(TokenizationError, match='pad_sequences'):
        tok(['ACGT', 'AC'])


def test_unbatched_leaves_ragged_input_ragged(tok):
    out = tok(['ACGT', 'AC'], batched=False)
    assert isinstance(out['input_ids'], list)
    assert [t.shape[0] for t in out['input_ids']] == [4, 2]
    assert all(t.dim() == 1 for t in out['input_ids'])


# -- sides ---------------------------------------------------------------------------


def test_right_padding_puts_the_sequence_first(tok):
    out = pad_sequences(tok(['ACGT', 'AC'], batched=False), side='right')
    assert out['input_ids'][1].tolist() == [0, 1, PAD_INDEX, PAD_INDEX]
    assert out['attention_mask'][1].tolist() == [1, 1, 0, 0]


def test_left_padding_puts_the_sequence_last(tok):
    out = pad_sequences(tok(['ACGT', 'AC'], batched=False), side='left')
    assert out['input_ids'][1].tolist() == [PAD_INDEX, PAD_INDEX, 0, 1]
    assert out['attention_mask'][1].tolist() == [0, 0, 1, 1]


def test_centre_tie_break_is_ceil_left():
    """The 600 nt source tiles were built ceil-left, floor-right; follow the data.

    parnet.data.datasets.ListDataset uses the opposite tie-break. For odd total padding
    the two differ by one nucleotide, which is a real shift in a position-sensitive model.
    """
    assert pad_sizes(length=2, target=5, side='center') == (2, 1)
    assert pad_sizes(length=3, target=6, side='center') == (2, 1)
    assert pad_sizes(length=2, target=6, side='center') == (2, 2), 'even padding is symmetric'


def test_centre_padding_on_an_odd_deficit(tok):
    out = pad_sequences(tok(['ACGTA', 'AC'], batched=False), side='center')
    assert out['input_ids'][1].tolist() == [PAD_INDEX, PAD_INDEX, 0, 1, PAD_INDEX]
    assert out['attention_mask'][1].tolist() == [0, 0, 1, 1, 0]


# -- the dataset layer's integer codes -----------------------------------------------


def test_pad_side_codes_match_the_dataset_layer():
    """meta['pad_side']: 0=center, 1=left, 2=right, -1=none."""
    assert PAD_SIDE_CODES == {0: 'center', 1: 'left', 2: 'right', -1: None}


@pytest.mark.parametrize(
    ('code', 'expected'),
    [(0, (2, 1)), (1, (3, 0)), (2, (0, 3)), (-1, (0, 0))],
)
def test_codes_and_names_agree(code, expected):
    assert pad_sizes(length=2, target=5, side=code) == expected
    assert pad_sizes(length=2, target=5, side=PAD_SIDE_CODES[code]) == expected


def test_a_bool_is_not_a_pad_side_code():
    """True would otherwise silently mean code 1, i.e. left."""
    with pytest.raises(TokenizationError, match='pad_side code'):
        pad_sizes(length=2, target=5, side=True)


def test_unknown_code_is_rejected():
    with pytest.raises(TokenizationError, match='unknown pad_side code'):
        pad_sizes(length=2, target=5, side=7)


# -- per-sequence sides ---------------------------------------------------------------


def test_each_sequence_can_have_its_own_side(tok):
    out = pad_sequences(tok(['AC', 'AC', 'AC'], batched=False), side=['left', 'right', 'center'], length=4)
    assert out['attention_mask'].tolist() == [[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 1, 0]]


def test_wrong_number_of_sides_is_rejected(tok):
    with pytest.raises(TokenizationError, match='one side or one per sequence'):
        pad_sequences(tok(['AC', 'AC'], batched=False), side=['left'])


# -- targets --------------------------------------------------------------------------


def test_default_target_is_the_longest(tok):
    out = pad_sequences(tok(['ACGTA', 'AC'], batched=False))
    assert out['input_ids'].shape == (2, 5)


def test_explicit_target_longer_than_every_sequence(tok):
    out = pad_sequences(tok(['ACGT', 'AC'], batched=False), side='right', length=10)
    assert out['input_ids'].shape == (2, 10)
    assert out['attention_mask'].sum().item() == 6


def test_padding_never_truncates(tok):
    with pytest.raises(TokenizationError, match='does not truncate'):
        pad_sequences(tok(['ACGTACGT'], batched=False), length=4)


def test_side_none_with_a_longer_target_is_an_error(tok):
    """None means 'do not pad', which cannot also mean 'reach this length'."""
    with pytest.raises(TokenizationError, match='cannot be combined'):
        pad_sequences(tok(['ACGT', 'AC'], batched=False), side=None)


# -- round trip -----------------------------------------------------------------------


def test_padding_is_invisible_to_the_model(tok):
    """PAD_INDEX is N, an all-zero one-hot column, so it adds nothing to a convolution."""
    from parnet.tokenization import ids_to_onehot

    out = pad_sequences(tok(['ACGT', 'AC'], batched=False), side='right')
    oh = ids_to_onehot(out['input_ids'])
    assert torch.equal(oh[1, :, 2:], torch.zeros(4, 2))
    assert oh[1, :, :2].sum() == 2
