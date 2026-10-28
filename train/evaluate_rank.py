# Constrained baseline: pick the label whose tokens have the highest log-likelihood after the prompt.
# Removes format-following from the comparison. Usage: evaluate_rank.py [adapter_dir]
import sys, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from common import *
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32); tag = "base"
if len(sys.argv) > 1:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, sys.argv[1]); tag = "lora"
model.eval(); _, test = splits(); ok = 0
for r in test:
    p = tok(PROMPT.format(text=r["text"]))["input_ids"]; best = None
    for L in LABELS:
        a = tok(" " + L)["input_ids"]; ids = torch.tensor([p + a])
        with torch.no_grad(): lp = torch.log_softmax(model(ids).logits[0, len(p) - 1:-1], -1)
        s = sum(lp[i, t].item() for i, t in enumerate(a))
        if best is None or s > best[0]: best = (s, L)
    ok += best[1] == r["label"]
print(json.dumps({"model": tag, "mode": "label log-likelihood ranking", "n": len(test), "correct": ok, "accuracy": round(ok / len(test), 3)}))
