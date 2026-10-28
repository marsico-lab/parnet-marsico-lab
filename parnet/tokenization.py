"""Tokenization of nucleotide sequences for PARNET.

The tokenizer is the first thing a user touches, so it is deliberately explicit about
the two ways real input differs from the idealised ``ACGU`` string, rather than silently
coercing them:

**Case.** Genomic FASTA is soft-masked: lowercase marks a repeat region, so it is real
sequence carrying a real annotation, not a formatting accident. Earlier code mapped every
lowercase base to an all-zero vector, which silently blanked roughly 30 per cent of
positions in MANE transcripts and produced predictions for sequence the model never saw.
Lowercase is therefore upper-cased, with one warning.

**Alphabet.** The models were trained on ``T``-bearing genomic sequence, while the
literature and most RNA tooling write ``U``. Both are accepted and map to the same
channel. A sequence containing *both* is rejected, because that almost always means two
sources were concatenated.

Everything else is an error that names the offending positions, because the failures this
module exists to prevent were all caused by something being quietly replaced by zeros.
"""

from __future__ import annotations

import warnings

import torch

__all__ = ['RnaTokenizer', 'ids_to_onehot', 'TokenizationError']

#: Channel order. ``T`` and ``U`` share channel 3; ``N`` is channel 4 and one-hots to zeros.
VOCAB: dict[str, int] = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4}

#: Number of real channels the model consumes. ``N`` is represented as an all-zero column.
N_CHANNELS = 4

#: Padding has its own id, distinct from ``N``.
#:
#: Both one-hot to the same all-zero column, so the distinction is invisible to the model
#: and cannot be otherwise: the released models take four input channels, and a fifth would
#: be a different architecture.
#:
#: What it buys is that ``input_ids`` describes itself. ``ids == PAD_INDEX`` recovers which
#: positions were invented without a second tensor travelling alongside, and the remaining
#: ids still decode to the original sequence, so a padded run can be compared against its
#: unpadded original from the ids alone.
#:
#: The distinction survives only on the ids. Hand the model a one-hot ``(B, 4, L)`` and it
#: is already gone, because padding and ``N`` are the same column there.
PAD_INDEX = 5

#: Reverse of :data:`VOCAB`, for :meth:`RnaTokenizer.decode`. ``T`` is not represented:
#: channel 3 decodes to ``U``, since the two share an id and RNA is the model's domain.
ID_TO_CHAR: dict[int, str] = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 4: 'N', PAD_INDEX: '-'}


class TokenizationError(ValueError):
    """Raised when input cannot be tokenized without guessing what was meant."""


class RnaTokenizer:
    """Turn nucleotide strings into integer ids.

    Args:
        warn_on_lowercase: Emit a :class:`UserWarning` the first time a lowercase base is
            seen. Set ``False`` only when you have already established that lowercase is
            expected, for example when deliberately feeding soft-masked genome.

    Example:
        >>> tok = RnaTokenizer()
        >>> batch = tok(["ACGU", "ACGT"])
        >>> batch["input_ids"].tolist()
        [[0, 1, 2, 3], [0, 1, 2, 3]]
    """

    def __init__(self, *, warn_on_lowercase: bool = True) -> None:
        self.warn_on_lowercase = warn_on_lowercase

    # -- single sequence -------------------------------------------------------------
    def encode(self, sequence: str) -> list[int]:
        """Encode one sequence to a list of ids.

        Raises:
            TokenizationError: if the sequence mixes ``T`` and ``U``, or contains a
                character outside ``ACGTUN`` (case-insensitive). The message names the
                offending positions, 0-based.
        """
        if not isinstance(sequence, str):
            raise TokenizationError(f'expected str, got {type(sequence).__name__}')

        had_lowercase = any(c.islower() for c in sequence)
        if had_lowercase and self.warn_on_lowercase:
            warnings.warn(
                'Lowercase bases were upper-cased. In genomic FASTA lowercase marks '
                'soft-masked repeats, which is real sequence; it is not dropped. Pass '
                'warn_on_lowercase=False to silence this.',
                UserWarning,
                stacklevel=2,
            )
        seq = sequence.upper()

        has_t = 'T' in seq
        has_u = 'U' in seq
        if has_t and has_u:
            t_at = [i for i, c in enumerate(seq) if c == 'T'][:5]
            u_at = [i for i, c in enumerate(seq) if c == 'U'][:5]
            raise TokenizationError(
                'sequence contains both T and U, which usually means two sources were '
                f'concatenated. T at {t_at}, U at {u_at} (first 5 of each shown). '
                'Convert the whole sequence to one alphabet before tokenizing.'
            )

        bad = [(i, c) for i, c in enumerate(seq) if c not in VOCAB]
        if bad:
            shown = bad[:10]
            raise TokenizationError(
                f'{len(bad)} character(s) outside ACGTUN. First offenders (position, '
                f'character): {shown}. IUPAC ambiguity codes and gaps are not accepted; '
                'resolve or mask them to N explicitly, so the choice is yours and visible.'
            )
        return [VOCAB[c] for c in seq]

    # -- batch -----------------------------------------------------------------------
    def __call__(
        self,
        sequences: str | list[str],
        *,
        return_tensors: str | None = 'pt',
        batched: bool = True,
    ) -> dict[str, torch.Tensor | list]:
        """Encode one or more sequences.

        This tokenizer does not pad. Where a short sequence sits inside a window is a
        modelling choice, not a tokenization detail, and in a position-sensitive model it
        moves every prediction; see :func:`parnet.padding.pad_sequences`.

        Args:
            sequences: A single sequence or a list of them.
            return_tensors: ``"pt"`` for ``torch.long`` tensors, ``None`` for plain lists.
            batched: ``True`` stacks the sequences into one ``(B, L)`` tensor and requires
                them to be the same length. ``False`` returns a list of 1-D tensors, one
                per sequence, leaving ragged input ragged.

        Returns:
            ``{"input_ids": ...}`` — a ``(B, L)`` tensor when ``batched``, otherwise a
            list of 1-D tensors. No ``attention_mask``: nothing here invents positions, so
            every position returned is real.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        if len(sequences) == 0:
            raise TokenizationError('no sequences given')

        ids = [self.encode(s) for s in sequences]

        if return_tensors is None:
            return {'input_ids': ids}
        if return_tensors != 'pt':
            raise TokenizationError(f"return_tensors must be 'pt' or None, got {return_tensors!r}")

        if not batched:
            return {'input_ids': [torch.tensor(x, dtype=torch.long) for x in ids]}

        lengths = {len(x) for x in ids}
        if len(lengths) > 1:
            raise TokenizationError(
                f'sequences have different lengths {sorted(lengths)} and batched=True. '
                'Either pass batched=False to get a list of 1-D tensors, or pad them '
                'yourself with parnet.padding.pad_sequences, which makes you say on which '
                'side the padding goes.'
            )
        return {'input_ids': torch.tensor(ids, dtype=torch.long)}

    def decode(self, input_ids: torch.Tensor | list[int], *, keep_padding: bool = False) -> str:
        """Turn ids back into a sequence.

        Together with the dedicated padding id this makes a padded batch reversible: the
        original sequence comes back without having been kept alongside.

        Args:
            input_ids: Ids for ONE sequence, as a 1-D tensor or a list.
            keep_padding: Render padded positions as ``-`` instead of dropping them.

        Returns:
            The sequence in ``ACGUN``. ``T`` does not come back: it shares an id with ``U``,
            and that is the one piece of input the tokenizer does not preserve.

        Raises:
            TokenizationError: If an id is outside the vocabulary.
        """
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() != 1:
                raise TokenizationError(f'decode takes one sequence; got shape {tuple(input_ids.shape)}')
            input_ids = input_ids.tolist()
        try:
            chars = [ID_TO_CHAR[int(i)] for i in input_ids]
        except KeyError as error:
            raise TokenizationError(f'id {error.args[0]} is not in the vocabulary') from None
        return ''.join(chars) if keep_padding else ''.join(c for c in chars if c != '-')

    def to_onehot(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert ids to the ``(B, 4, L)`` one-hot the model consumes.

        The third of the three steps, alongside :meth:`encode` and :meth:`decode`. This is
        where ``N`` and padding stop being distinguishable, since both become the all-zero
        column; read anything you need off the ids first.

        Args:
            input_ids: Ids of shape ``(batch, length)``.

        Returns:
            Float one-hot of shape ``(batch, 4, length)``.

        Raises:
            TokenizationError: If ``input_ids`` is not 2-D.
        """
        return ids_to_onehot(input_ids)


def ids_to_onehot(input_ids: torch.Tensor) -> torch.Tensor:
    """Convert ``(B, L)`` integer ids to the ``(B, 4, L)`` float one-hot the model consumes.

    Both ``N`` and padding become all-zero columns. This is where the two stop being
    distinguishable, so anything that needs to know which positions were padding must read
    it off the ids before calling this, or let the model do it by passing ``input_ids``.

    An all-zero column is inert in the first convolution only. Biases and batch-norm give
    the padded region activations downstream, so padding is not free; see
    :mod:`parnet.padding`.
    """
    if input_ids.dim() != 2:
        raise TokenizationError(f'expected (B, L) ids, got shape {tuple(input_ids.shape)}')
    oh = torch.nn.functional.one_hot(input_ids.long(), num_classes=PAD_INDEX + 1)
    return oh[..., :N_CHANNELS].permute(0, 2, 1).to(torch.float32)
