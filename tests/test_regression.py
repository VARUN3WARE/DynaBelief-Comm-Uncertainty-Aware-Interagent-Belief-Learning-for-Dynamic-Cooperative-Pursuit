"""Frozen confirmatory arms must train bit-identically at every later commit.

The reference hashes in ``tests/data/regression_hashes.json`` were recorded at the
freeze (``scripts/regression_hash.py --all``). Floating-point results legitimately
differ across CPU models and torch versions, so the check only runs where the
reference was recorded; regenerate references only at a declared freeze, never to make
this test pass.
"""

from __future__ import annotations

import json

import pytest
import torch
from scripts.regression_hash import cpu_model, fingerprint

from tests.helpers import REPO_ROOT

REFS = json.loads((REPO_ROOT / "tests" / "data" / "regression_hashes.json").read_text())


@pytest.mark.slow
@pytest.mark.parametrize("ref", REFS, ids=[r["config"] for r in REFS])
def test_frozen_arm_trains_bit_identically(ref):
    if torch.__version__ != ref["torch"] or cpu_model() != ref["cpu_model"]:
        pytest.skip(f"reference recorded on {ref['cpu_model']} / torch {ref['torch']}")
    got = fingerprint(ref["config"], ref["updates"], ref["device"], ref["threads"])
    assert got["losses"] == ref["losses"]
    assert got["sha256"] == ref["sha256"]
