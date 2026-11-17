# The measured results the Worker serves (data/results.json) must be self-consistent.
import json, math
from pathlib import Path

RESULTS = json.loads((Path(__file__).resolve().parents[1] / "data" / "results.json").read_text())


def test_same_heldout_split_for_both_models():
    assert RESULTS["before"]["n"] == RESULTS["after"]["n"] == RESULTS["split"]["heldout_test"]


def test_accuracy_matches_correct_over_n():
    for m in (RESULTS["before"], RESULTS["after"]):
        assert m["accuracy"] == round(m["correct"] / m["n"], 3)
        assert 0 <= m["accuracy"] <= 1 and m["correct"] <= m["n"]


def test_delta_is_after_minus_before():
    assert RESULTS["delta_accuracy"] == round(RESULTS["after"]["accuracy"] - RESULTS["before"]["accuracy"], 3)


def test_training_log_records_lora_config_and_finite_loss():
    t = RESULTS["training"]
    assert t["trainable_params"] < t["total_params"] / 50 and len(t["loss_log"]) >= 5
    for x in t["loss_log"]:
        assert math.isfinite(x if isinstance(x, (int, float)) else x["loss"])
