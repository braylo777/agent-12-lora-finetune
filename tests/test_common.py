# Unit tests for the training/eval helpers (no model download) and for the measured results data.
import json, math, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "train"))
import pytest

pytest.importorskip("datasets", reason="train/common.py imports `datasets` (pip install datasets)")
from common import parse, LABELS, PROMPT  # noqa: E402

RESULTS = json.loads((root / "data" / "results.json").read_text())


def test_parse_exact():
    for L in LABELS:
        assert parse(" " + L) == L


def test_parse_prefix_and_case():
    assert parse(" sports\n") == "Sports" and parse(" Tech news") == "Sci/Tech"


def test_parse_rejects_noise():
    assert parse(" The headline") is None


def test_prompt_has_all_labels():
    assert all(L in PROMPT for L in LABELS)
