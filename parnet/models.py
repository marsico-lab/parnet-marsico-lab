import gin
import torch
import torch.nn as nn
import captum


from parnet.utils import sequence_to_onehot
from parnet.layers import StemConv1D, ResConvBlock1D, AdditiveMix, _LegacyStateMixin


@gin.configurable()
class RBPNet(_LegacyStateMixin, nn.Module):
    """Implements the RBPNet model as described in Horlacher et al. (2023), DOI: https://doi.org/10.1186/s13059-023-03015-7."""

    def __init__(
        self,
        num_tasks: int = None,
        layers: int = 9,
        dilation: int = 1.75,
        stem_layer: nn.Module = StemConv1D,
        body_layer: nn.Module = ResConvBlock1D,
        head_layer: nn.Module = AdditiveMix,
        embedding_dim: int = 128,
        use_projection: bool = True,
    ):
        """Initializes RBPNet.

        Args:
            num_tasks (int): Number of tasks (i.e. eCLIP tracks).
            layers (int, optional): Number of body layer, e.g. residual blocks. Defaults to 9.
            dilation (float, optional): Dilation coeff. for convolutions in the body layers. The i'th body layer will have a coeff. of floor(dilation**i). Defaults to 1.75.
            head_layer (nn.Module, optional): Layer to use for the output head. Defaults to LinearProjection.
            use_projection (bool, optional): Whether to apply the linear projection layer
                between body and head. When False, the head reads the body's own output
                directly (identity in place of the projection conv). Previously only
                achievable by monkey-patching model.projection after construction (see
                parnet_additional_utils.load_parnet_model's bypass_projection, used for
                legacy checkpoints predating this layer) -- exposed here as a real
                constructor/gin argument instead. Defaults to True.
        """
        super().__init__()

        if num_tasks is None:
            # We could infer this from the dataset, but let's keep it explicit for now.
            raise ValueError('num_tasks must be specified in the gin config file.')

        self.stem = stem_layer()
        self.body = nn.Sequential(*[body_layer(dilation=int(dilation**i)) for i in range(layers)])

        # final linear projection layer to produce sequence embeddings
        if use_projection:
            self.projection = nn.LazyConv1d(embedding_dim, kernel_size=1, bias=False, padding='same')
        else:
            self.projection = nn.Identity()

        if head_layer is None:
            raise ValueError('head_layer must be specified.')
        self.head = head_layer(num_tasks)

        # Dummy forward pass to initialize weights. Not strictly required, but allows us
        # to print a proper summary of the model with pytorch_lightning and get the correct
        # number of parameters.
        _ = self(torch.zeros(2, 4, 100, dtype=torch.float32))

    def forward(
        self,
        sequence: torch.Tensor = None,
        to_probs=False,
        *,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """Performs a forward pass through the model, returning logits for each postion and task.

        Args:
            sequence (torch.Tensor): Batched one-hot encoded sequences of shape (batch_size, num_channels, sequence_length).
            to_probs (bool, optional): Whether to convert logits to probabilities. Defaults to False.

        Returns:
            dict[str, torch.Tensor]: A dictionary logits for total, target, control tracks and the mixing coefficient.
        """
        if sequence is None and input_ids is None:
            raise ValueError('pass either a one-hot `sequence` (B, 4, L) or `input_ids` (B, L)')
        if sequence is not None and input_ids is not None:
            raise ValueError('pass `sequence` or `input_ids`, not both')
        if sequence is None:
            sequence = input_ids
        # `attention_mask` is accepted for API symmetry with the tokenizer's output and is
        # deliberately unused: padding is encoded as N, an all-zero column, which already
        # contributes nothing to any convolution.
        del attention_mask

        # Accept either the (B, 4, L) float one-hot this model has always taken, or the
        # (B, L) integer ids produced by parnet.tokenization.RnaTokenizer, so that
        # `model(**tokenizer(seqs))` works without the caller one-hotting by hand. Ids are
        # distinguished by rank and dtype, never by value, so a 2-D float tensor is still
        # rejected downstream rather than silently reinterpreted.
        if sequence.dim() == 2 and not torch.is_floating_point(sequence):
            from parnet.tokenization import ids_to_onehot

            sequence = ids_to_onehot(sequence)

        x = self.stem(sequence)
        x = self.body(x)
        # `projection` is None when the model has none, which is the case for every
        # currently released checkpoint. Guarding here keeps the knowledge of whether a
        # projection exists in the model rather than in a config that could disagree
        # with the weights.
        if self.projection is not None:
            x = self.projection(x)

        x = self.head(x)

        if isinstance(x, torch.Tensor):
            x = {'total': x}

        if to_probs:
            # convert track logits to probabilities
            x['total'] = x['total'].softmax(dim=-1)
            if 'target' in x:
                x['target'] = x['target'].softmax(dim=-1)
            if 'control' in x:
                x['control'] = x['control'].softmax(dim=-1)
        return x

    @classmethod
    def from_pretrained(cls, stem, *, map_location='cpu', strict: bool = True):
        """Load a model saved as ``<stem>.safetensors`` + ``<stem>.config.json``.

        Returns the model in eval mode. See :mod:`parnet.pretrained` for the format and
        for why the weights, not the config, decide whether a projection exists.
        """
        from parnet.pretrained import from_pretrained as _from_pretrained

        return _from_pretrained(stem, map_location=map_location, strict=strict)

    def save_pretrained(self, stem, *, name=None, extra=None):
        """Write this model as ``<stem>.safetensors`` + ``<stem>.config.json``."""
        from parnet.pretrained import save_pretrained as _save_pretrained

        return _save_pretrained(self, stem, name=name, extra=extra)

    def predict_from_sequence(self, sequences: str | list[str], to_probs=False):
        """Predicts RBP binding probabilities from a sequence.

        Args:
            sequence (str): Sequence(s) to predict from.
            to_probs (bool, optional): Whether to convert logits to probabilities. Defaults to False.

        Returns:
            torch.Tensor: Predicted binding probabilities.
        """

        if isinstance(sequences, str):
            sequences = [sequences]

        # one-hot encode sequence(s)
        sequence_onehot = torch.stack([sequence_to_onehot(s) for s in sequences]).float()

        return self.forward(sequence_onehot, to_probs=to_probs)

    def embed(self, sequence: torch.Tensor, aggregation: str = None):
        """Returns sequence embeddings.

        Args:
            sequence (torch.Tensor): Batched one-hot encoded sequences.
            aggregation (str, optional): One of 'mean' or 'center'. Defaults to None.
        """
        x = self.stem(sequence)
        x = self.body(x)
        # Same guard as forward(): `projection` is None for checkpoints that have none,
        # which is every released model loaded from a legacy pickle. Calling None(x)
        # here would raise TypeError from inside embed() with no useful message.
        if self.projection is not None:
            x = self.projection(x)

        if aggregation is None:
            return x

        if aggregation == 'mean':
            return x.mean(dim=-1)
        elif aggregation == 'center':
            return x[:, :, x.shape[-1] // 2]
        else:
            raise ValueError('aggregation must be one of "mean", "center" or None.')

    def embed_from_sequence(self, sequences: str | list[str], aggregation: str = None):
        if isinstance(sequences, str):
            sequences = [sequences]

        # one-hot encode sequence(s)
        sequence_onehot = torch.stack([sequence_to_onehot(s) for s in sequences]).float()

        return self.embed(sequence_onehot, aggregation=aggregation)

    def explain(self, sequence: torch.Tensor, task_idx: int, track='target'):
        """Returns attribution maps for the given task and track.

        Args:
            sequence (torch.Tensor): Batched one-hot encoded sequences of shape (batch_size, num_channels, sequence_length).
            task_idx (int): Index of the task to explain.
            track (str, optional): Track to explain. Defaults to 'target'.
        """
        # NOTE: In the future we should support explaination for all tasks and tracks at once.

        def _explain_forward(inputs):
            pred = self.forward(inputs)[track][:, task_idx, :].softmax(dim=-1)
            return (pred * pred.detach()).sum(dim=-1)

        return captum.attr.InputXGradient(_explain_forward).attribute(sequence)

    def explain_from_sequence(self, sequences, task_idx: int, track='target'):
        if isinstance(sequences, str):
            sequences = [sequences]

        # one-hot encode sequence(s)
        sequence_onehot = torch.stack([sequence_to_onehot(s) for s in sequences]).float()

        return self.explain(sequence_onehot, task_idx, track=track)
