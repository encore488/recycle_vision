"""The Colab notebook has to agree with the current data decisions.

It went stale without anyone noticing: it still trained on WaRP months after
WaRP was retired from training, and it cloned the default branch, so a run
started from it would have used neither the pool nor any of the guards. A
notebook is documentation that executes, and it rots the same way documentation
does — except that following it costs GPU hours.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "train_colab.ipynb"


@pytest.fixture(scope="module")
def source() -> str:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells)


@pytest.fixture(scope="module")
def code(source: str) -> str:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def test_the_notebook_is_valid(source: str):
    assert json.loads(NOTEBOOK.read_text(encoding="utf-8"))["nbformat"] == 4


def test_it_trains_on_the_pool(code: str):
    """WaRP is retired from training; a notebook that trains on it is wrong."""
    train = [line for line in code.splitlines() if "train.py" in line]
    assert train, "no training command at all"
    assert "--data datasets/pool/data.yaml" in code


def test_it_does_not_train_on_warp(code: str):
    """The exact stale command this test was written to catch."""
    assert "train.py --data datasets/warp" not in code.replace("\\\n", "")
    for line in code.splitlines():
        if "train.py" in line:
            assert "warp" not in line.lower()


def test_it_evaluates_on_warp_as_a_holdout(code: str):
    assert "datasets/warp/data.yaml" in code
    assert "evaluate.py" in code


def test_it_gates_on_preflight_before_training(code: str):
    """Preflight after training would cost exactly what it exists to save."""
    assert "preflight.py" in code
    assert code.index("preflight.py") < code.index("train.py")


def test_preflight_failure_stops_the_notebook(code: str):
    """A printed warning nobody reads is not a gate."""
    assert "returncode == 0" in code or "check=True" in code


def test_it_checks_out_an_explicit_branch(code: str):
    """Cloning the default branch silently omits every fix not yet merged."""
    assert "git checkout" in code
    assert "BRANCH" in code


def test_it_rewrites_the_descriptor_path(code: str):
    """Trap 5: data.yaml carries the absolute path of the machine that built it."""
    assert 'spec["path"]' in code


def test_it_clears_the_label_caches(code: str):
    """Trap 2: the cache is keyed on byte size and paths, never contents."""
    assert ".cache" in code


def test_it_saves_weights_to_drive(code: str):
    """Colab reclaims the machine and takes the run with it."""
    assert "drive" in code.lower()
    assert "best_model.pt" in code


def test_it_checks_whether_the_run_finished(code: str):
    """Three runs died; a dead run leaves what a converged one leaves."""
    assert "inspect_run" in code


def test_it_scopes_the_holdout_classes(code: str):
    assert "--classes present" in code


def test_it_asserts_a_gpu_is_present(code: str):
    """Without the accelerator the whole notebook runs on CPU for nothing."""
    assert "cuda.is_available" in code


def test_it_names_the_number_to_beat(source: str):
    """A run with no target produces a number nobody can judge."""
    assert "16.2%" in source
