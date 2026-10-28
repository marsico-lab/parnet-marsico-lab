# PARNET

Per-nucleotide prediction of RNA-binding protein signal from sequence.

PARNET predicts three tracks for each task: `target`, the RBP eCLIP signal; `control`, the
size-matched input background; and `total`, a learned combination of the two. 
Each track is a density over positions i.e. the model predicts where signal falls rather than absolute read counts.

## Install

```bash
pip install git+https://github.com/marsico-lab/parnet-marsico-lab.git@refactor
```

On a machine with an older NVIDIA driver, install torch from the matching CUDA index **first**. 
Otherwise pip silently resolves a build whose `torch.cuda.is_available()` is `False`.

```bash
pip install --index-url https://download.pytorch.org/whl/cu118 "torch>=2.5"
pip install git+https://github.com/marsico-lab/parnet-marsico-lab.git@refactor
```

## Quick start

```python
import torch
import parnet
from parnet.models import RBPNet

tokenizer = parnet.RnaTokenizer()
model = RBPNet.from_pretrained('path/to/parnet-21m-penalty5')

seq = 'ACGUACGU' * 75  # 600 nt, the window the models were trained on
encoded = tokenizer([seq, seq])  # (2, 600); a bare string would give (1, 600)

with torch.no_grad():
    out = model(**encoded, to_probs=True)

out['target']  # (batch, tasks, length)  probability over positions, sums to 1
out['control']  # (batch, tasks, length)  probability over positions, sums to 1
out['total']  # (batch, tasks, length)  the learned combination
out['mix_coeff']  # (batch, tasks)         raw output of the mixing layer
out['target_weight']  # (batch, tasks)         w in [0, 1], the share carried by target
```

Drop `to_probs=True` to get log-probabilities instead.
That is the better scale for anything that compares profiles, and it is what the training loss consumes.
Without it, `total` from a log-space-mixed model such as the 21M carries the mixing coefficient as an additive offset, so it is a shape without a scale and `exp()` overflows.

`from_pretrained` takes a **stem**; it appends `.safetensors` for the weights and
`.config.json` for the architecture, and returns the model in eval mode.

### Sequences of different lengths

The tokenizer does not pad, because where a short sequence sits inside the window changes
every prediction and is not a default worth guessing.
Ask for the ragged form, then pad, choosing the side:

```python
from parnet.padding import pad_sequences

ragged = tokenizer([seq, seq[:400]], batched=False)  # list of 1-D tensors, lengths 600 and 400
padded = pad_sequences(ragged, side='center')  # 'left', 'right', or a pad_side code

with torch.no_grad():
    out = model(**padded, to_probs=True)
```

Padding has its own token id, so `padded['input_ids']` records which positions were
invented and nothing has to be passed alongside it.
`to_probs` then normalises over the real positions only, and `tokenizer.decode` gives the
original sequences back.

Nothing is truncated.
Stride is 1 and padding is `same` throughout, so output length equals input length.

Padding a short sequence into a longer window is **not** free, however.
An all-zero input column is inert only in the first convolution: after that, biases and batch-norm make the padded region carry activations of the same magnitude as real sequence, and the dilated stack propagates them back.
Padding a 300 nt sequence to 600 shifts the 21M's embedding by up to 0.55 at every position, including the one furthest from the padding, because the receptive field is wider than the window.
Pad as little as possible, keep to the 600 nt the models were trained on, and do not compare profiles computed at different padded lengths.


## Model files

A model is `<name>.safetensors` plus `<name>.config.json`. Whole-module pickles are
deprecated: a pickle stores a class path and executes code on load, so a renamed class or
an attribute added after the file was written breaks it, far from the cause.

`config.json` carries the architecture, including the parts that are **not** in the state
dict: activations, kernel sizes, dropout and the dilation schedule. 


## Input

`RnaTokenizer` accepts DNA or RNA, upper or lower case, and is explicit about everything
else:

```python
tokenizer = parnet.RnaTokenizer()
tokenizer('ACGT')  # same ids as 'acgu'
```

Anything outside `ACGTUN` raises `TokenizationError`, naming the offending positions.

The tokenizer provides an argument `batched=True`, that stacks a list of sequences into one `(batch, length)` tensor.
This requires the sequences to be the same length. 
`batched=False` returns a list of 1-D tensors and leaves ragged input ragged:

```python
tokenizer(['ACGU', 'GGCC'])  # (2, 4) tensor
tokenizer(['ACGU', 'GG'], batched=False)  # [tensor(4), tensor(2)]
```

Padding lives in `parnet.padding`, where the side has to be stated:

```python
from parnet.padding import pad_sequences

encoded = tokenizer(['ACGU', 'GG'], batched=False)
pad_sequences(encoded, side='center')  # also 'left', 'right', or one side per sequence
```

It accepts the dataset layer's integer `pad_side` codes (`0` center, `1` left, `2` right,
`-1` none), always returns an `attention_mask` so no caller has to infer which positions
were invented, and refuses to truncate. 

The center tie-break is ceil-left, floor-right, matching the 600 nt source tiles; this is checked against the dataset layer's own
`_compute_pad_sizes`. 

`forward` accepts either the tokenizer's `input_ids` of shape `(batch, length)` or a one-hot `(batch, 4, length)`.

## The output head

Two mixing rules exist and they are not interchangeable.

- **convex combination**, `w * p_target + (1 - w) * p_control`, needs `w` in `[0, 1]`
- **log-space addition**, `logsumexp(w + log p_target, log p_control)`, takes an unbounded logit

Which one runs is set by `use_maximum_target_control_logprob`. A mixing layer declares what
it emits in `mix_coeff_range`, and the constructor refuses an incompatible pairing, because
one direction gives a silent error and the other a `NaN` far from its cause. A bounded
coefficient fed to the log-sum rule is squashed twice, confining the effective target
weight to `(0.5, 0.731)` so the control track can never dominate.

The penalty acts on `w`, never on the raw layer output. For a logit-emitting layer those
are different objectives: penalising `w` is bounded in `[0, factor]` with a gradient that
vanishes as `w` approaches 0, while penalising the logit is unbounded below with a constant
gradient and never stops pushing.

## Embeddings

```python
with torch.no_grad():
    full = model.embed(x)  # (batch, channels, length)
    per_sequence = model.embed(x, aggregation='mean')  # or 'center'
```

`embed` returns what the output head would have received, so it accounts for a projection
if the model has one.

## Fine-tuning

The backbone is reusable; the head is what you replace. Swap `model.head` and leave the
body and projection alone, because the projection is part of the learned representation.

Choose what to train by **stages that own parameters**, not by counting layers:

```python
from parnet.finetuning import stage_names, set_trainable, describe_trainable

stage_names(model)  # ['stem', 'body.0', ..., 'body.8']
set_trainable(model, unfreeze_last=0)  # head only
describe_trainable(model)
```

"Unfreeze the last N layers" breaks as soon as a model may or may not have a trainable
projection, because the last trainable thing is then sometimes `body[-1]` and sometimes
`projection`. A parameterless projection never appears in `stage_names`, and
never consumes a slot, so the same call means the same thing on every model.

## Seeding

```python
import parnet  # imports first

parnet.set_seed(42)  # then seed
```

Importing parnet does not reseed your RNG and does not change global numerics. It used to
do both, via `seed_everything(42)` and `set_float32_matmul_precision('high')`; the latter
is now `parnet.use_tensor_cores()`.

Seed **after** your imports. Importing `lightning` consumes random numbers by itself and
parnet imports it transitively, so seeding first and importing afterwards leaves you on a
different stream.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest tests -q
ruff check . && ruff format --check .
```

The pre-commit hooks run the ruff from your environment rather than pinning their own, so
the hook and your terminal cannot disagree about whether a file is clean. CI installs the
same dev extra.
