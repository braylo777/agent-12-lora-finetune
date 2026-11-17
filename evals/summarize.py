"""Builds evals/results.json and data/results.json from the raw eval outputs.
Picks paired examples where the two models differ."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parent
rd = lambda f: json.loads((here / f).read_text())
before, after, train = rd("before.json"), rd("after.json"), rd("train_log.json")
rank = rd("before_rank.json") if (here / "before_rank.json").exists() else {"status": "not yet run"}
strip = lambda m: {k: v for k, v in m.items() if k != "rows"}
pairs = [{"text": b["text"], "gold": b["gold"],
          "before": b["pred"] if b.get("pred") is not None else f'(unparseable: "{b["raw"].strip()}")',
          "after": after["rows"][i]["pred"] if after["rows"][i].get("pred") is not None else f'(unparseable: "{after["rows"][i]["raw"].strip()}")',
          "before_ok": b["ok"], "after_ok": after["rows"][i]["ok"]} for i, b in enumerate(before["rows"])]
out = {"project": "agent-12-lora-finetune", "task": "AG News topic classification (4 classes), generated label, exact match", "dataset": "fancyzhx/ag_news",
       "split": {"train": train["train_examples"], "heldout_test": before["n"], "note": "held-out = 200 shuffled rows (seed 0) of the official AG News test split"},
       "base_model": train["base"], "training": strip(train), "before": strip(before), "after": strip(after), "base_constrained_ranking": rank,
       "delta_accuracy": round(after["accuracy"] - before["accuracy"], 3),
       "examples": [*[p for p in pairs if not p["before_ok"] and p["after_ok"]][:6], *[p for p in pairs if not p["after_ok"]][:3]],
       "workers_ai_live_serving": {"status": "not available", "reason": "Workers AI only loads LoRA adapters onto its own hosted base models (e.g. Gemma, Mistral, Llama), and SmolLM2-135M is not one of them, so the demo serves measured results only."}}
(root / "data").mkdir(exist_ok=True)
(here / "results.json").write_text(json.dumps({**out, "rows_before": before["rows"], "rows_after": after["rows"]}, indent=2))
(root / "data" / "results.json").write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
print(json.dumps({"before": strip(before), "after": strip(after), "rank": rank, "delta": out["delta_accuracy"], "train_s": train["train_seconds"]}, indent=1))
