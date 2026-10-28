import re, json, random
from datasets import load_dataset
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
PROMPT = "Classify the news headline into one topic: World, Sports, Business, or Sci/Tech.\nHeadline: {text}\nTopic:"
BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"
def splits(n_train=2000, n_test=200, seed=0):
    ds = load_dataset("fancyzhx/ag_news")
    tr = ds["train"].shuffle(seed=seed).select(range(n_train))
    te = ds["test"].shuffle(seed=seed).select(range(n_test))  # held-out: AG News official test split, never trained on
    f = lambda r: {"text": r["text"][:300], "label": LABELS[r["label"]]}
    return [f(r) for r in tr], [f(r) for r in te]
def parse(out):
    s = out.strip().lower()
    for L in LABELS:
        if s.startswith(L.lower()) or (L == "Sci/Tech" and s.startswith(("sci", "tech"))): return L
    return None
