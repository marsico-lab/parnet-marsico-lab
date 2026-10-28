"""Padding has its own id, and the model can read it."""

import torch

from parnet.padding import pad_sequences
from parnet.tokenization import N_CHANNELS, PAD_INDEX, RnaTokenizer, ids_to_onehot


def test_padding_is_not_n():
    """The two mean different things, so they are different ids."""
    assert PAD_INDEX != RnaTokenizer().encode('N')[0]
    assert PAD_INDEX >= N_CHANNELS


def test_padding_and_n_are_the_same_column_to_the_model():
    """They must be: the released models take four input channels."""
    oh = ids_to_onehot(torch.tensor([[0, 4, PAD_INDEX]]))
    assert oh.shape == (1, 4, 3)
    assert torch.equal(oh[0, :, 1], torch.zeros(4)), 'N'
    assert torch.equal(oh[0, :, 2], torch.zeros(4)), 'padding'


def test_ids_say_which_positions_were_invented():
    tok = RnaTokenizer()
    out = pad_sequences(tok(['ACGU', 'AC'], batched=False), side='right')
    derived = out['input_ids'] != PAD_INDEX
    assert torch.equal(derived, out['attention_mask'].bool()), 'the mask must be recoverable from the ids alone'


def test_decode_returns_the_original_sequence():
    tok = RnaTokenizer()
    padded = pad_sequences(tok(['ACGUACGU', 'ACG'], batched=False), side='center')
    assert tok.decode(padded['input_ids'][0]) == 'ACGUACGU'
    assert tok.decode(padded['input_ids'][1]) == 'ACG', 'padding dropped, original recovered'


def test_decode_can_show_the_padding():
    tok = RnaTokenizer()
    padded = pad_sequences(tok(['ACGU', 'AC'], batched=False), side='left')
    assert tok.decode(padded['input_ids'][1], keep_padding=True) == '--AC'


def test_decode_does_not_restore_t():
    """T and U share an id; U is what comes back, and that is the only loss."""
    tok = RnaTokenizer()
    assert tok.decode(tok('ACGT')['input_ids'][0]) == 'ACGU'


def test_to_probs_spends_nothing_on_padding():
    """A distribution over positions should not include positions that do not exist."""
    import functools

    from parnet import layers
    from parnet.models import RBPNet

    torch.manual_seed(0)
    model = RBPNet(
        num_tasks=2,
        layers=2,
        embedding_dim=8,
        use_projection=False,
        head_layer=functools.partial(
            layers.TargetControlMix,
            mix_coeff_layer=layers.PooledProbMixCoeff,
            use_maximum_target_control_logprob=True,  # convex rule takes a unit-range coefficient
        ),
    )
    model.eval()

    tok = RnaTokenizer()
    padded = pad_sequences(tok(['ACGUACGUAC', 'ACGU'], batched=False), side='right')
    keep = padded['input_ids'] != PAD_INDEX

    with torch.no_grad():
        out = model(input_ids=padded['input_ids'], to_probs=True)
    probs = out['target']
    on_pad = probs.masked_fill(keep.unsqueeze(1), 0.0).sum()
    assert float(on_pad) == 0.0, 'padded positions must receive no probability'
    real_sum = probs.masked_fill(~keep.unsqueeze(1), 0.0).sum(-1)
    assert torch.allclose(real_sum, torch.ones_like(real_sum), atol=1e-5)

    with torch.no_grad():
        unmasked = model(input_ids=padded['input_ids'], to_probs=True, mask_padding=False)
    spent = unmasked['target'].masked_fill(keep.unsqueeze(1), 0.0).sum()
    assert float(spent) > 0.0, 'mask_padding=False must reproduce the old whole-window softmax'
