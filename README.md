# parnet

Sequence models of protein-RNA binding. `parnet` predicts per-nucleotide eCLIP signal from
RNA sequence, and provides the pieces needed to use a pretrained model: a tokenizer, the
model, embeddings, and parametrised freezing for fine-tuning.

## Install

```bash
pip install -e .
```

That is the whole thing on a machine with a recent NVIDIA driver. **On an older driver it
will silently give you a CPU-only build**, so check first:

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

| driver | what to do |
|---|---|
| 525 or newer | `pip install -e .` and nothing else |
| 450 to 524 | install torch from the CUDA 11.8 index first, then parnet |

```bash
# older driver: pin the CUDA 11.8 build FIRST, then install parnet normally
pip install --index-url https://download.pytorch.org/whl/cu118 "torch>=2.5"
pip install -e .
```

The second command leaves the pinned torch alone, because `torch >=2.5` is already
satisfied. Verified on driver 515.105.01: after step one, `torch 2.7.1+cu118` with
`cuda.is_available()` True; after step two, still `2.7.1+cu118`, still True, both GPUs
visible.

This matters because the default wheel on PyPI is built against a recent CUDA. On the same
machine a plain `pip install -e .` pulled `torch 2.14.1+cu130` and
`torch.cuda.is_available()` returned **False** with two idle A6000s in the box. There is no
error in that case, only silence and slow training, which is why this section is first.

`uv` works the same way and is faster:

```bash
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -e .
```

Dependencies are deliberately unpinned above a floor. If you need a lockfile, generate one
for your environment rather than expecting the library to impose one.

## Quick start

```python
import torch
from parnet import RnaTokenizer

model = torch.load("parnet.21m-5.0.pt", weights_only=False)
model.eval()                                    # see "Eval mode" below, this is not optional

tok = RnaTokenizer()
out = model(**tok(["ACGUACGU...", "acgtacgt..."]))

out["total"]      # (batch, tasks, length) mixed signal logits
out["target"]     # (batch, tasks, length) target track
out["control"]    # (batch, tasks, length) control track
out["mix_coeff"]  # (batch, tasks) mixing coefficient
```

### Input

`RnaTokenizer` takes DNA or RNA, upper or lower case:

- `T` and `U` both map to the same channel. A sequence containing **both** is rejected,
  because that usually means two sources were concatenated.
- Lowercase is upper-cased with one warning. In genomic FASTA lowercase marks soft-masked
  repeats, which is real sequence. It is never dropped.
- `N` is allowed and becomes an all-zero column.
- Anything else, including IUPAC ambiguity codes and gaps, is an error naming the
  positions. Resolve or mask them to `N` yourself, so the choice is visible.
- Sequences of unequal length are padded with `N`, which is inert because the model is
  fully convolutional.

`model(sequence=...)` with a `(batch, 4, length)` float one-hot still works unchanged.

## Eval mode

**Call `.eval()` before predicting.** Some released checkpoints were saved in training
mode, and the models contain `BatchNorm1d` and `Dropout1d`. Without `.eval()` you get batch
statistics instead of running statistics and active dropout, so predictions are both wrong
and different on every call. Measured on a 3M checkpoint: train and eval mode differ by
4.88 in logit space, and two successive forward passes in train mode are not identical.

## Models

Three families, with different widths. Do not hardcode these numbers; read
`model.body[-1].conv1d.out_channels`, or the length of an embedding.

| family | parameters | embedding width | tasks | head |
|---|---|---|---|---|
| 21M | 21,814,495 | 768 | 223 | `AdditiveMix` with `SequenceLinearMix` |
| 7M | 7,534,815 | 512 | 223 | `NewAdditiveMix` with `NewMixCoeffMLP` |
| 3M | 2,870,067 | 320 | varies by split | `AdditiveMix` with `MixCoeffMLP` |

### The mixing coefficient

`mix_coeff` is not the same quantity in every family, which is the single most common way
to misread a prediction.

- With `SequenceLinearMix` (the 21M) it is an **unbounded logit**, and the target weight is
  `sigmoid(mix_coeff)`. Values in the thousands are normal.
- With `MixCoeffMLP` or `NewMixCoeffMLP` (the 7M and 3M) the layer already applies a
  sigmoid, so `mix_coeff` **is** the weight, in [0, 1].

Pairing a layer that emits one with a mixing rule that expects the other produces NaN or a
weight silently confined to (0.5, 0.731). The constructor now refuses such a pair with
`IncompatibleMixError`; pass `strict_mix=False` to override.

## Embeddings

```python
h = model.body(model.stem(x))     # (batch, channels, length)
```

`model.projection` sits between body and head. No currently released model has a real one,
so the embedding width is the body width. The released files disagree on how they spell
that: the 21M has no `projection` attribute at all, the 7M has one holding `None`, and the
3M backbones have one holding `nn.Identity`. All three now behave identically, because
loading fills a missing attribute with `None` and `forward` skips a `None` projection.

A future model may carry a real projection, in which case the head sees its output instead,
and `model.projection is not None` is how you tell.

## Fine-tuning

Freezing is parametrised by **stages that own parameters**, not by layers, so the same call
means the same thing whether or not a model has a trainable projection:

```python
from parnet import set_trainable, stage_names

stage_names(model)                      # ['stem', 'body.0', ..., 'body.8']
report = set_trainable(model, unfreeze_last=2)
print(report)                           # which stages are trainable, and parameter counts
```

`unfreeze_last=0` freezes the whole backbone, `-1` unfreezes all of it, `N` leaves the last
`N` stages trainable. A parameterless projection is not a stage, so it never consumes a
slot. Asking for more stages than exist is an error rather than a silent clamp.

To swap the head, replace `model.head` and leave the backbone and projection alone: the
projection is part of the learned representation.

## Shared machines

`torch` defaults to one thread per core, so several people training at once on one box
oversubscribe it badly. Measured with four processes on 16 cores: throughput per process
fell from 47 to 3.9 steps per second, and aggregate throughput fell too. Capping threads
fixed it, restoring 34.3 steps per second each. Caps of 1, 2 and 4 threads all performed
the same; 8 dropped to 23.6 and the uncapped 16 to 3.9.

```python
import torch
torch.set_num_threads(2)
```

Put that at the top of a shared notebook. GPU and host memory were not the constraint:
264 MiB of GPU per process and 1.4 GiB of RAM.

## Tests

```bash
pip install pytest && pytest tests -q
```

One test module is skipped unless `tensorflow` and `tensorflow_probability` are present.
They are used only as an independent reference for the multinomial log-probability and are
not dependencies of the library.
