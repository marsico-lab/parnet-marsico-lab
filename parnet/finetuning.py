"""Parametrised freezing and unfreezing for fine-tuning.

The usual phrasing, "unfreeze the last N layers of the body", stops working as soon as a
model may or may not have a trainable projection between body and head: the last trainable
thing is sometimes ``body[-1]`` and sometimes ``projection``, and writing either in a
notebook silently does the wrong thing on the other kind of model.

The fix is to stop counting *layers* and start counting *stages that own parameters*. A
module with no parameters, such as the ``nn.Identity`` standing in for an absent
projection, is not a stage and never consumes an unfreeze slot. "Unfreeze the last two
stages" then means the same thing on every model, and a notebook written against it keeps
working when a projection appears.
"""

from __future__ import annotations

import torch.nn as nn

__all__ = ["trainable_stages", "stage_names", "set_trainable", "describe_trainable"]


def _owns_parameters(module: nn.Module) -> bool:
    return any(p.numel() > 0 for p in module.parameters(recurse=True))


def trainable_stages(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Ordered backbone stages that own parameters, input side first.

    The head is deliberately excluded: it is what you replace, not what you unfreeze a
    part of. Stages are ``stem``, then each residual block as ``body.<i>``, then
    ``projection`` when one exists and carries weights.

    Returns:
        A list of ``(name, module)``. ``len()`` of it is the number of unfreezable stages,
        so ``unfreeze_last=len(stages)`` is the same as unfreezing the whole backbone.
    """
    stages: list[tuple[str, nn.Module]] = []
    stem = getattr(model, "stem", None)
    if stem is not None and _owns_parameters(stem):
        stages.append(("stem", stem))
    body = getattr(model, "body", None)
    if body is not None:
        for i, block in enumerate(body):
            if _owns_parameters(block):
                stages.append((f"body.{i}", block))
    projection = getattr(model, "projection", None)
    # None (no projection at all) and nn.Identity (a placeholder) both own no parameters,
    # so neither becomes a stage. Only a real projection does.
    if projection is not None and _owns_parameters(projection):
        stages.append(("projection", projection))
    return stages


def stage_names(model: nn.Module) -> list[str]:
    """Just the names, for printing in a notebook before choosing ``unfreeze_last``."""
    return [name for name, _ in trainable_stages(model)]


def set_trainable(
    model: nn.Module,
    *,
    unfreeze_last: int = 0,
    head: bool = True,
) -> dict[str, object]:
    """Freeze the backbone and selectively unfreeze its last stages.

    Args:
        model: An ``RBPNet``-shaped model, with ``stem``, ``body`` and optionally
            ``projection`` and ``head``.
        unfreeze_last: How many backbone stages, counted from the output side, to leave
            trainable. ``0`` freezes the whole backbone, which is the usual
            feature-extraction setup. ``-1`` unfreezes all of it, equivalent to full
            fine-tuning. A value larger than the number of stages is an error rather than
            a silent clamp, because it usually means the caller assumed a different
            architecture.
        head: Whether ``model.head`` stays trainable. Normally yes, since the head is the
            new part.

    Returns:
        A dict describing what happened: ``stages`` (all stage names), ``unfrozen``
        (the stage names left trainable), ``n_trainable`` and ``n_frozen`` parameter
        counts, and ``head_trainable``. Print it in a notebook rather than trusting that
        the argument did what you meant.
    """
    stages = trainable_stages(model)
    n = len(stages)
    if unfreeze_last < -1:
        raise ValueError(f"unfreeze_last must be -1, 0 or positive, got {unfreeze_last}")
    if unfreeze_last > n:
        raise ValueError(
            f"unfreeze_last={unfreeze_last} but this model has {n} trainable backbone "
            f"stages: {[s for s, _ in stages]}. Pass -1 for all of them."
        )

    keep = n if unfreeze_last == -1 else unfreeze_last
    unfrozen_names = [name for name, _ in stages[n - keep :]] if keep else []

    for name, module in stages:
        requires = name in unfrozen_names
        for p in module.parameters(recurse=True):
            p.requires_grad_(requires)

    head_module = getattr(model, "head", None)
    if head_module is not None:
        for p in head_module.parameters(recurse=True):
            p.requires_grad_(bool(head))

    return describe_trainable(model) | {
        "stages": [name for name, _ in stages],
        "unfrozen": unfrozen_names,
        "head_trainable": bool(head) if head_module is not None else None,
    }


def describe_trainable(model: nn.Module) -> dict[str, int]:
    """Parameter counts, as a cheap sanity check after calling :func:`set_trainable`."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"n_trainable": trainable, "n_frozen": total - trainable, "n_total": total}
