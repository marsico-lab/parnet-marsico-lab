__version__ = '0.1.1'

# Register only the external classes parnet actually uses. gin's own
# gin.torch.external_configurables is deliberately NOT imported: it registers
# torch.nn wholesale, including the deprecated NLLLoss2d, which emits a
# FutureWarning on every import of parnet.
from . import _gin_external_configurables

# add tensor cores support
import torch

torch.set_float32_matmul_precision('high')

# Set random seeds
from lightning.pytorch import seed_everything

__seed__ = 42
seed_everything(__seed__)

from . import models, layers, metrics, losses, data, tokenization
from .tokenization import RnaTokenizer, ids_to_onehot
from .data import datasets
