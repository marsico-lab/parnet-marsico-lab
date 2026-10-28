"""Schema for a model's ``config.json``.

The architecture block is validated STRICTLY (``extra='forbid'``): it is what rebuilds a
model, so a typo there must fail loudly at load rather than be ignored and produce a
model that is quietly not the one on disk. Everything else, meaning the provenance and
reference blocks, is permissive, because those grow over time and an unknown key in them
is a note, not a hazard.

This replaces ad-hoc key lookups in :mod:`parnet.pretrained` with one declared shape, and
it is the first piece of the move off gin: model construction no longer needs a
configuration framework at all, only this schema plus the weights.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)


class _Open(BaseModel):
    model_config = ConfigDict(extra='allow')


class StemConfig(_Strict):
    """The stem convolution. ``filters`` is not stored here; it is ``body_width``."""

    class_name: str = Field(alias='class')
    kernel_size: int
    activation: str
    dropout: float | None = None


class BodyConfig(_Strict):
    """The residual stack.

    ``dilations`` is the per-block schedule as built, not the base from which it was
    derived, because the schedule is what must be reproduced and the base cannot be
    recovered from it unambiguously.
    """

    class_name: str = Field(alias='class')
    kernel_size: int
    activation: str
    dropout: float | None = None
    residual: bool = True
    dilations: list[int]


class ArchitectureConfig(_Strict):
    """Everything needed to rebuild the model, including the non-parametric parts.

    Activations, kernel sizes, dropout and the dilation schedule do not appear in a state
    dict. A config that omits them yields a model that loads cleanly with ``strict=True``
    and computes something different; that has happened.
    """

    class_name: Literal['RBPNet'] = Field(alias='class')
    n_body_blocks: int
    body_width: int
    embedding_dim: int
    use_projection: bool
    stem: StemConfig
    body: BodyConfig
    mix_coeff_units: int | None = None
    mix_coeff_activation: str | None = None
    head_class: str
    mix_coeff_class: str
    mix_coeff_range: Literal['unit', 'logit'] | None = None
    head_layer_class: str
    use_maximum_target_control_logprob: bool
    control_nograd: bool
    penalty_class: str | None = None
    penalty_factor: float | None = None
    lambda_anchor: float | None = None
    n_parameters: int
    n_tasks: int


class UnverifiableClaim(_Open):
    """A fact about the training run that cannot be measured from the file.

    The released 21M is the case this exists for: it was trained at lambda 5.0, but the
    checkpoint carries ``penalty = None``, so the value survives only in the filename.
    Keeping it here rather than in ``architecture`` stops it being mistaken for something
    read off the weights.
    """

    quantity: str
    value: Any
    source: str
    recoverable_from_weights: bool = False
    note: str | None = None


class ModelConfig(_Open):
    """A whole ``config.json``."""

    name: str
    architecture: ArchitectureConfig
    export: dict[str, Any] = Field(default_factory=dict)
    reference: dict[str, Any] = Field(default_factory=dict)
    unverifiable_claims: list[UnverifiableClaim] = Field(default_factory=list)
