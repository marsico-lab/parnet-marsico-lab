# PARNET

Per-nucleotide prediction of RNA-binding protein signal from sequence.

PARNET predicts three tracks for each task: `target`, the RBP eCLIP signal; `control`, the
size-matched input background; and `total`, a learned combination of the two. Each track is
a density over positions, so the model predicts where signal falls, not how much there is.

## Install

```bash
pip install git+https://github.com/marsico-lab/parnet-marsico-lab.git@refactor
```

On a machine with an older NVIDIA driver, install torch from the matching CUDA index
**first**. Otherwise pip resolves a build whose `torch.cuda.is_available()` is `False`,
silently, with two idle GPUs and no error anywhere:

```bash
pip install --index-url https://download.pytorch.org/whl/cu118 "torch>=2.5"
pip install git+https://github.com/marsico-lab/parnet-marsico-lab.git@refactor
```

The second command leaves the pin intact, because `torch>=2.5` is already satisfied.

### Version bounds

parnet is a library, so it does not cap its dependencies speculatively. An upper bound
set here propagates into every downstream resolution and causes conflicts the consumer
cannot fix. `requires-python` is `>=3.10` rather than a single minor version, and the
only real floor is `torch>=2.5`. If you hit a genuine incompatibility, a cap with a
comment naming it is welcome.

## Quick start

```python
import torch
import parnet
from parnet.models import RBPNet

tokenizer = parnet.RnaTokenizer()
model = RBPNet.from_pretrained('path/to/parnet-21m-penalty5')

with torch.no_grad():
    out = model(**tokenizer('ACGUACGU' * 75))

out['target']        # (batch, tasks, length)  log-density over positions
out['control']       # (batch, tasks, length)
out['total']         # (batch, tasks, length)  the learned combination
out['mix_coeff']     # (batch, tasks)          raw output of the mixing layer
out['target_weight'] # (batch, tasks)          w in [0, 1], the share carried by target
```

`from_pretrained` takes a **stem**; it appends `.safetensors` for the weights and
`.config.json` for the architecture, and returns the model in eval mode.

Use `target_weight`, not `mix_coeff`, when you want the mixing weight. Whether
`mix_coeff` already is `w` depends on which mixing layer the model has, and deciding that
by hand has gone wrong in both directions.

## Model files

A model is `<name>.safetensors` plus `<name>.config.json`. Whole-module pickles are
deprecated: a pickle stores a class path and executes code on load, so a renamed class or
an attribute added after the file was written breaks it, far from the cause.

`config.json` carries the architecture, including the parts that are **not** in the state
dict: activations, kernel sizes, dropout and the dilation schedule. Omitting those
produces a model that loads cleanly with `strict=True` and computes something different,
which has happened: a set of backbones trained with GELU were rebuilt with the default
ReLU and nobody noticed until their stored reference outputs failed to reproduce.

The weights are the source of truth. `use_projection` in a config is derived metadata, and
loading recomputes it from the state dict and raises `ConfigWeightMismatch` if the two
disagree, rather than trusting either.

Each config also records a reference measurement: the mean target weight on a seeded input
set, with the seed and sequence count. Reproducing it is a cheap check that a model is the
one you think it is, and it is sensitive at once to a wrong formula flag, a double sigmoid,
a lost penalty and a reduction mismatch. Use that seeded set rather than your own draw; at
eight sequences, two independent draws differ by about ten per cent.

## Input

`RnaTokenizer` accepts DNA or RNA, upper or lower case, and is explicit about everything
else:

```python
tokenizer = parnet.RnaTokenizer()
tokenizer('ACGT')          # same ids as 'acgu'
```

Anything outside `ACGTUN` raises `TokenizationError`, naming the offending positions.
IUPAC ambiguity codes and gaps are included in that: resolve or mask them to `N` yourself.

This matters more than it sounds. The older helper, `sequence_to_onehot`, silently turns a
lowercase sequence into an all-zero tensor and the model still returns a confident
prediction. Soft-masked genome sequence is lowercase, so that was reachable from ordinary
input.

Sequences of unequal length are currently right-padded with `N`, which is inert because
padding is an all-zero column and the model is fully convolutional. That is a batching
artifact, not a modelling choice, and the tokenizer is the wrong place for it: the dataset
layer already pads using each window's recorded `pad_side` (`left`, `right`, or centre for
tiles built from genes shorter than the window). Padding policy is moving out of the
tokenizer.

`forward` accepts either the tokenizer's `input_ids` of shape `(batch, length)` or a
one-hot `(batch, 4, length)`.

### Sequence length

Nothing fixes the input at 600 nt: every layer is `stride=1, padding='same'` and there is
no upsampling, so any length runs. What differs between models is the receptive field,
computed from the recorded dilation schedule. Densities are normalised per window, so they
are not comparable across lengths; compare shapes.

## The output head

Two mixing rules exist and they are not interchangeable.

- **convex combination**, `w * p_target + (1 - w) * p_control`, needs `w` in `[0, 1]`
- **log-space addition**, `logsumexp(w + log p_target, log p_control)`, takes an unbounded
  logit

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
    full = model.embed(x)                        # (batch, channels, length)
    per_sequence = model.embed(x, aggregation='mean')    # or 'center'
```

`embed` returns what the output head would have received, so it accounts for a projection
if the model has one. Ask for the width rather than hard-coding it.

Use **cosine similarity**, not Euclidean distance. The body is a plain residual stack with
no renormalisation after the residual add, so activation norm grows with depth and any
unnormalised metric picks that up as if it were signal.

## Fine-tuning

The backbone is reusable; the head is what you replace. Swap `model.head` and leave the
body and projection alone, because the projection is part of the learned representation.

Choose what to train by **stages that own parameters**, not by counting layers:

```python
from parnet.finetuning import stage_names, set_trainable, describe_trainable

stage_names(model)              # ['stem', 'body.0', ..., 'body.8']
set_trainable(model, unfreeze_last=0)   # head only
describe_trainable(model)
```

"Unfreeze the last N layers" breaks as soon as a model may or may not have a trainable
projection, because the last trainable thing is then sometimes `body[-1]` and sometimes
`projection`. A parameterless projection owns nothing, never appears in `stage_names`, and
never consumes a slot, so the same call means the same thing on every model.

## Seeding

```python
import parnet          # imports first
parnet.set_seed(42)    # then seed
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
