"""Tests for parnet.tokenization.

Each test corresponds to a rule agreed with the user on 2026-10-08. The rules exist
because of specific past failures, named in the comments, so a test that starts failing
should be read as "that failure can happen again".
"""

import pytest
import torch

from parnet.tokenization import (
    N_CHANNELS,
    RnaTokenizer,
    TokenizationError,
    ids_to_onehot,
)
from parnet.utils import sequence_to_onehot


@pytest.fixture
def tok():
    return RnaTokenizer()


def test_acgu_maps_to_channels_0_to_3(tok):
    assert tok.encode('ACGU') == [0, 1, 2, 3]


def test_dna_t_maps_to_the_same_channel_as_u(tok):
    # The models were trained on T-bearing genomic sequence; U is the literature's spelling.
    assert tok.encode('ACGT') == tok.encode('ACGU')


def test_lowercase_is_uppercased_not_zeroed(tok):
    # Regression: lowercase marks soft-masked repeats in genomic FASTA. Zeroing it blanked
    # ~30% of positions in MANE transcripts and produced predictions for unseen sequence.
    with pytest.warns(UserWarning, match='soft-masked'):
        ids = tok.encode('acgt')
    assert ids == [0, 1, 2, 3]


def test_lowercase_warning_can_be_disabled():
    silent = RnaTokenizer(warn_on_lowercase=False)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any warning becomes an error
        assert silent.encode('acgt') == [0, 1, 2, 3]


def test_mixed_t_and_u_is_a_hard_error(tok):
    # Almost always two sources concatenated, so guessing an alphabet would hide a data bug.
    with pytest.raises(TokenizationError, match='both T and U'):
        tok.encode('ACGTACGU')


def test_n_is_allowed_and_one_hots_to_zeros(tok):
    ids = tok.encode('ACGN')
    assert ids == [0, 1, 2, 4]
    oh = ids_to_onehot(torch.tensor([ids]))
    assert oh[0, :, 3].sum().item() == 0.0


def test_iupac_codes_are_rejected_with_positions(tok):
    with pytest.raises(TokenizationError) as e:
        tok.encode('ACGRYT')
    msg = str(e.value)
    assert 'outside ACGTUN' in msg
    assert "(3, 'R')" in msg and "(4, 'Y')" in msg


def test_gap_characters_are_rejected(tok):
    with pytest.raises(TokenizationError, match='outside ACGTUN'):
        tok.encode('AC-GT')


def test_batch_of_equal_length_has_no_attention_mask(tok):
    out = tok(['ACGT', 'UGCA'])
    assert out['input_ids'].shape == (2, 4)
    assert 'attention_mask' not in out  # absence means every position is real


def test_unequal_lengths_are_refused(tok):
    """The tokenizer does not pad; padding is parnet.padding's job and must be asked for."""
    with pytest.raises(TokenizationError, match='different lengths'):
        tok(['ACGT', 'AC'])


def test_ragged_output_is_available_without_padding(tok):
    out = tok(['ACGT', 'AC'], batched=False)
    assert [t.shape[0] for t in out['input_ids']] == [4, 2]
    assert 'attention_mask' not in out


def test_equal_lengths_still_stack(tok):
    out = tok(['ACGT', 'UGCA'])
    assert out['input_ids'].shape == (2, 4)
    oh = ids_to_onehot(out['input_ids'])
    assert oh.shape == (2, 4, 4)
    assert float(oh.sum()) == 8.0, 'every position is a real base'


def test_single_string_is_accepted(tok):
    assert tok('ACGT')['input_ids'].shape == (1, 4)


def test_empty_batch_is_an_error(tok):
    with pytest.raises(TokenizationError, match='no sequences'):
        tok([])


def test_onehot_shape_and_dtype(tok):
    oh = ids_to_onehot(tok(['ACGT'])['input_ids'])
    assert oh.shape == (1, N_CHANNELS, 4)
    assert oh.dtype == torch.float32


def test_matches_legacy_sequence_to_onehot_on_uppercase_acgt(tok):
    # Nothing changes for input that the old path already handled correctly, so existing
    # results stay reproducible.
    seq = 'ACGTACGTTTGGCCAA'
    new = ids_to_onehot(tok([seq])['input_ids'])[0]
    old = sequence_to_onehot(seq).float()
    assert torch.equal(new, old)


def test_differs_from_legacy_exactly_where_legacy_was_wrong(tok):
    # U and lowercase are the two cases the old function silently zeroed.
    for seq in ['ACGU', 'acgt']:
        old = sequence_to_onehot(seq).float()
        new = ids_to_onehot(RnaTokenizer(warn_on_lowercase=False)([seq])['input_ids'])[0]
        assert not torch.equal(new, old)
        assert new.sum() > old.sum()  # the old path lost information


def test_to_onehot_is_a_tokenizer_method():
    """encode, decode and to_onehot are the three steps, on one object."""
    tok = RnaTokenizer()
    ids = tok(['ACGU'])['input_ids']
    assert torch.equal(tok.to_onehot(ids), ids_to_onehot(ids))
    assert tok.to_onehot(ids).shape == (1, 4, 4)
