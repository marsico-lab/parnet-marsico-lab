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

#: Index used for padding; identical to ``N`` because the model is fully convolutional and
#: an all-zero column contributes nothing to any convolution.
PAD_INDEX = 4


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


def ids_to_onehot(input_ids: torch.Tensor) -> torch.Tensor:
    """Convert ``(B, L)`` integer ids to the ``(B, 4, L)`` float one-hot the model consumes.

    ``N`` and padding (index 4) become all-zero columns, which contribute nothing to any
    convolution.
    """
    if input_ids.dim() != 2:
        raise TokenizationError(f'expected (B, L) ids, got shape {tuple(input_ids.shape)}')
    oh = torch.nn.functional.one_hot(input_ids.long(), num_classes=N_CHANNELS + 1)
    return oh[..., :N_CHANNELS].permute(0, 2, 1).to(torch.float32)
