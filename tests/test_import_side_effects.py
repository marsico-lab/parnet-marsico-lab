"""Importing parnet must not change global state.

Run in a subprocess, because the only honest way to test "import has no side effects" is
to observe a fresh interpreter.
"""

import subprocess
import sys


def _run(code):
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip()


def test_import_does_not_change_matmul_precision():
    """set_float32_matmul_precision is process-wide and changes numerics everywhere."""
    output = _run(
        'import torch;'
        'before = torch.get_float32_matmul_precision();'
        'import parnet;'
        'print(before, torch.get_float32_matmul_precision())'
    )
    before, after = output.split()
    assert before == after, f'importing parnet changed matmul precision {before} -> {after}'


def test_import_does_not_reseed_the_global_rng():
    """parnet must not force the caller's RNG onto its own seed.

    Note what this does NOT claim. Importing ``lightning`` consumes random numbers on its
    own, and every parnet submodule imports lightning transitively, so the global stream
    is perturbed no matter what parnet does. That is upstream and not parnet's to fix.

    What parnet is responsible for is not RESEEDING, which is what ``seed_everything(42)``
    at import time did: it made the caller's stream deterministic and identical for
    everyone, silently overriding any seed they had set. This asserts the state after
    importing parnet is not the seed-42 state.
    """
    seeded = _run(
        'from lightning.pytorch import seed_everything; seed_everything(42);'
        'import random; print(random.random())'
    )
    after_import = _run('import random; random.seed(1234); import parnet; print(random.random())')
    assert after_import != seeded, (
        'importing parnet reset the global RNG to its own seed; seeding must be explicit'
    )


def test_seeding_is_available_explicitly():
    output = _run('import parnet; parnet.set_seed(7); import random; print(random.random())')
    again = _run('import parnet; parnet.set_seed(7); import random; print(random.random())')
    assert output == again


def test_tensor_cores_opt_in_still_works():
    output = _run(
        'import torch, parnet;'
        'parnet.use_tensor_cores("high");'
        'print(torch.get_float32_matmul_precision())'
    )
    assert output == 'high'
