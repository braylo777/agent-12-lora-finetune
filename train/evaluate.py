# Exact-match accuracy on the held-out split, greedy decoding. Usage: evaluate.py [adapter_dir] > out.json
import sys, json, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from common import *
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(BASE); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
tag = "base"
if len(sys.argv) > 1:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, sys.argv[1]); tag = "lora"
model.eval(); _, test = splits()
rows, t0 = [], time.perf_counter()
for i in range(0, len(test), 16):
    b = test[i:i+16]; enc = tok([PROMPT.format(text=r["text"]) for r in b], return_tensors="pt", padding=True)
    with torch.no_grad(): g = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
    for r, o in zip(b, tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
        p = parse(o); rows.append({"text": r["text"][:120], "gold": r["label"], "raw": o, "pred": p, "ok": p == r["label"]})
n = len(rows); acc = sum(r["ok"] for r in rows) / n
per = {L: round(sum(r["ok"] for r in rows if r["gold"] == L) / max(1, sum(r["gold"] == L for r in rows)), 3) for L in LABELS}
print(json.dumps({"model": tag, "n": n, "accuracy": round(acc, 3), "correct": sum(r["ok"] for r in rows), "unparseable": sum(r["pred"] is None for r in rows), "per_class_accuracy": per, "ms_per_example": round((time.perf_counter() - t0) * 1000 / n, 1), "rows": rows}))
