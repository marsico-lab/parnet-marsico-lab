__version__ = '0.1.1'

# Register only the external classes parnet actually uses. gin's own
# gin.torch.external_configurables is deliberately NOT imported: it registers
# torch.nn wholesale, including the deprecated NLLLoss2d, which emits a
# FutureWarning on every import of parnet.
from . import _gin_external_configurables

import torch

#: Default seed, kept so callers can refer to the value parnet used to set implicitly.
DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, workers: bool = True) -> int:
    """Seed Python, numpy and torch.

    Importing a library must not reseed the caller's global RNG, so this is explicit.
    Until now ``import parnet`` called ``seed_everything(42)``, which silently reset
    the random state of whatever else was running in the same process and made any
    experiment interleaved with a parnet import irreproducible for reasons invisible
    at the call site.
    """
    from lightning.pytorch import seed_everything

    return seed_everything(seed, workers=workers)


def use_tensor_cores(precision: str = 'high') -> None:
    """Allow lower-precision float32 matmuls on tensor cores.

    This is a PROCESS-WIDE numerics change and it affects every other library in the
    same interpreter, so it is opt-in rather than an import side effect. ``import
    parnet`` used to call ``torch.set_float32_matmul_precision('high')`` on your behalf;
    if you were relying on that for throughput, call this once at the top of your script.
    """
    torch.set_float32_matmul_precision(precision)


from . import models, layers, metrics, losses, data, tokenization, finetuning
from .tokenization import RnaTokenizer, ids_to_onehot
from .finetuning import set_trainable, stage_names, trainable_stages
from .data import datasets
