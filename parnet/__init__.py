"""PARNET: per-nucleotide prediction of RBP binding from sequence."""

__version__ = '0.1.1'

# Register only the external classes parnet actually uses. gin's own
# gin.torch.external_configurables is deliberately NOT imported: it registers torch.nn
# wholesale, including the deprecated NLLLoss2d, which emits a FutureWarning on every
# import of parnet.
from . import _gin_external_configurables  # noqa: F401  (import registers gin configurables)

import torch

from . import data, finetuning, layers, losses, metrics, models, tokenization
from .data import datasets
from .finetuning import set_trainable, stage_names, trainable_stages
from .tokenization import RnaTokenizer, ids_to_onehot

#: Default seed, kept so callers can refer to the value parnet used to set implicitly.
DEFAULT_SEED = 42

__all__ = [
    'DEFAULT_SEED',
    'RnaTokenizer',
    'data',
    'datasets',
    'finetuning',
    'ids_to_onehot',
    'layers',
    'losses',
    'metrics',
    'models',
    'set_seed',
    'set_trainable',
    'stage_names',
    'tokenization',
    'trainable_stages',
    'use_tensor_cores',
]


def set_seed(seed: int = DEFAULT_SEED, workers: bool = True) -> int:
    """Seed Python, numpy and torch.

    Importing a library must not reseed the caller's global RNG, so this is explicit.
    ``import parnet`` used to call ``seed_everything(42)``, which silently reset the
    random state of whatever else was running in the same process.

    Call this AFTER your imports. Importing ``lightning`` consumes random numbers by
    itself, and parnet imports lightning transitively, so seeding first and importing
    afterwards leaves you on a different stream than seeding last. Measured: seeding
    then importing parnet gives a different first draw; importing then seeding is
    identical to seeding alone.

    Args:
        seed: Value to seed with.
        workers: Also seed dataloader workers.

    Returns:
        The seed that was set.
    """
    from lightning.pytorch import seed_everything

    return seed_everything(seed, workers=workers)


def use_tensor_cores(precision: str = 'high') -> None:
    """Allow lower-precision float32 matmuls on tensor cores.

    This is a PROCESS-WIDE numerics change affecting every other library in the same
    interpreter, so it is opt-in rather than an import side effect. ``import parnet`` used
    to call ``torch.set_float32_matmul_precision('high')`` on your behalf; if you relied
    on that for throughput, call this once at the top of your script.

    Args:
        precision: One of ``'highest'``, ``'high'`` or ``'medium'``.
    """
    torch.set_float32_matmul_precision(precision)
