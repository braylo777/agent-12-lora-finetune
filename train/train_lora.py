# LoRA fine-tune of SmolLM2-135M-Instruct on AG News topic labels, CPU only. Loss is on the answer tokens only.
import sys, json, time, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from common import *
torch.manual_seed(0); random.seed(0)
tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, cfg); trainable, total = model.get_nb_trainable_parameters()
train, _ = splits()
def batch(rs):
    ids, labs = [], []
    for r in rs:
        p = tok(PROMPT.format(text=r["text"]))["input_ids"]; a = tok(" " + r["label"])["input_ids"] + [tok.eos_token_id]
        ids.append(p + a); labs.append([-100] * len(p) + a)
    m = max(map(len, ids))
    return (torch.tensor([x + [tok.pad_token_id] * (m - len(x)) for x in ids]), torch.tensor([x + [-100] * (m - len(x)) for x in labs]), torch.tensor([[1] * len(x) + [0] * (m - len(x)) for x in ids]))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
BS, EPOCHS, log, t0 = 8, 1, [], time.perf_counter()
model.train()
for ep in range(EPOCHS):
    random.shuffle(train)
    for s in range(0, len(train), BS):
        x, y, m = batch(train[s:s+BS]); loss = model(input_ids=x, attention_mask=m, labels=y).loss
        loss.backward(); opt.step(); opt.zero_grad()
        step = s // BS + 1
        if step % 25 == 0: log.append({"step": step, "loss": round(loss.item(), 4)}); print(log[-1], flush=True)
model.save_pretrained("adapter")
json.dump({"base": BASE, "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "targets": cfg.target_modules and sorted(cfg.target_modules)}, "trainable_params": trainable, "total_params": total,
  "train_examples": len(train), "epochs": EPOCHS, "batch_size": BS, "lr": 3e-4, "train_seconds": round(time.perf_counter() - t0, 1), "hardware": "box CPU (8 vCPU)", "loss_log": log}, open("evals/train_log.json", "w"), indent=2)
