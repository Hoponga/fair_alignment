
import os
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets
import matplotlib 
import math
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------
# Load tokenizer & model (Llama-3 Instruct best practices)
# --------------------------
MODEL_PATH = "./models/llama3_8b_rm_fair/last_checkpoint"
MAX_LEN = 2048
BATCH_SIZE = 64
N_EVAL = 1000
SEED = 42

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
tokenizer.model_max_length = MAX_LEN

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    attn_implementation="sdpa",
)
model.to(device).eval()

# --------------------------
# Helpers: apply chat template & batch scoring
# --------------------------
def render_messages(messages):
    """messages is a list of {'role','content'} per RLHFlow; return templated string."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

@torch.inference_mode()
def score_texts(texts):
    """Return numpy array of reward scores for a list of templated strings."""
    out_scores = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i:i+BATCH_SIZE]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).to(device)
        logits = model(**enc).logits.squeeze(-1)  # [B]
        out_scores.append(logits.detach().float().cpu().numpy())
    return np.concatenate(out_scores, axis=0) if out_scores else np.array([])

# --------------------------
# Load datasets & build eval slice
# --------------------------
print("Loading RLHFlow HH datasets…")
ds_helpful  = load_dataset("RLHFlow/HH-RLHF-Helpful-standard", split="train")
ds_helpful  = ds_helpful.map(lambda ex: {"category": "helpful"})
ds_harmless = load_dataset("RLHFlow/HH-RLHF-Harmless-and-RedTeam-standard", split="train")
ds_harmless = ds_harmless.map(lambda ex: {"category": "harmless"})

ds_all = concatenate_datasets([ds_helpful, ds_harmless]).shuffle(seed=SEED)
n = min(N_EVAL, len(ds_all))
eval_ds = ds_all.select(range(n))

# Each example has keys: 'chosen' (messages list), 'rejected' (messages list), 'category'
print(f"Scoring {len(eval_ds)} pairs…")
chosen_txts   = [render_messages(ex["chosen"])   for ex in eval_ds]
rejected_txts = [render_messages(ex["rejected"]) for ex in eval_ds]
categories    = [ex["category"] for ex in eval_ds]

# --------------------------
# Score
# --------------------------
scores_chosen   = score_texts(chosen_txts)
scores_rejected = score_texts(rejected_txts)
assert scores_chosen.shape == scores_rejected.shape == (len(eval_ds),)

# Accuracy: chosen > rejected
acc = float(np.mean(scores_chosen > scores_rejected)) * 100.0
print(f"Accuracy (chosen > rejected): {acc:.2f}%")

# --------------------------
# Plots
# --------------------------
from scipy.stats import gaussian_kde

def _plot_kde(data, label, color=None):
    kde = gaussian_kde(data)
    xs = np.linspace(min(data), max(data), 300)
    ys = kde(xs)
    plt.plot(xs, ys, label=label, color=color, linewidth=2)
    plt.fill_between(xs, ys, alpha=0.3, color=color)

# 1) Chosen vs Rejected distribution
plt.figure(figsize=(7,5))
_plot_kde(scores_chosen,   "Chosen")
_plot_kde(scores_rejected, "Rejected")
plt.title("Reward Distribution: Chosen vs Rejected")
plt.xlabel("Reward score")
plt.ylabel("Density")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("dist_chosen_vs_rejected.png")
print("Saved: dist_chosen_vs_rejected_fair.png")

# 2) Helpful vs Harmless (both chosen & rejected pooled)
helpful_mask  = np.array([c == "helpful"  for c in categories], dtype=bool)
harmless_mask = np.array([c == "harmless" for c in categories], dtype=bool)

helpful_scores  = np.concatenate([scores_chosen[helpful_mask],  scores_rejected[helpful_mask]])
harmless_scores = np.concatenate([scores_chosen[harmless_mask], scores_rejected[harmless_mask]])

plt.figure(figsize=(7,5))
_plot_kde(helpful_scores,  "Helpful (chosen+rejected)")
_plot_kde(harmless_scores, "Harmless (chosen+rejected)")
plt.title("Reward Distribution: Helpful vs Harmless")
plt.xlabel("Reward score")
plt.ylabel("Density")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("dist_helpful_vs_harmless.png")
print("Saved: dist_helpful_vs_harmless_fair.png")

# --------------------------
# Summary
# --------------------------
print("\nSummary:")
print(f"  Pairs scored: {len(eval_ds)}")
print(f"  Accuracy (chosen > rejected): {acc:.2f}%")
print("  Figures:")
print("    - dist_chosen_vs_rejected.png")
print("    - dist_helpful_vs_harmless.png")