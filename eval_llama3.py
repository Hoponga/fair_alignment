import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------
# Config / device
# -------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Your finetuned reward-model checkpoint (must have a 1-logit head)
CHECKPOINT_PATH = "./models/llama3_8b_rm_fair/last_checkpoint"

# -------------------------
# Tokenizer & model (Llama 3 Instruct)
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, use_fast=False)
# causal LMs prefer left padding; Llama-3 uses </s> as EOS
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load with 1-logit head (num_labels=1 should already be in the checkpoint; harmless if repeated)
model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT_PATH,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
).to(device)
model.eval()

# -------------------------
# Chat template helper
# -------------------------
def format_chat(text: str) -> str:
    """
    Wrap plain text in Llama 3 Instruct chat template as a single-user turn.
    If your dataset already contains a formatted conversation, you can bypass this.
    """
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

# -------------------------
# Scoring functions
# -------------------------
@torch.inference_mode()
def score_single(text: str) -> float:
    """
    Score a single response with the reward head (returns a scalar float).
    """
    if isinstance(text, (list, tuple)):
        text = text[-1]
    prompt = format_chat(text)
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=tokenizer.model_max_length,
    ).to(device)
    logits = model(**enc).logits.squeeze(-1)  # [1] -> scalar
    return logits.item()

@torch.inference_mode()
def score_batch(texts):
    """
    Efficiently score a list of texts at once. Returns a 1D numpy array of floats.
    """
    prompts = [format_chat(t[-1] if isinstance(t, (list, tuple)) else t) for t in texts]
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,  # pad to longest in-batch
        max_length=tokenizer.model_max_length,
    ).to(device)
    logits = model(**enc).logits.squeeze(-1)  # [B, 1] -> [B]
    return logits.detach().float().cpu().numpy()

# -------------------------
# Dataset: RewardBench v2
# -------------------------
rb = load_dataset("allenai/reward-bench-2", "default", split="test")

# -------------------------
# Demo: sample raw scores (first 50)
# -------------------------
print("\nSample raw scores (first 50 examples):")
samples = rb.select(range(50))
for ex in samples:
    sc_chosen = score_single(ex["chosen"])
    # batch-score all rejected for speed
    sc_rej = score_batch(ex["rejected"])
    print(
        f" CHOSEN: {sc_chosen:.4f}   REJ max: {sc_rej.max():.4f}   "
        f"DIFF: {(sc_chosen - sc_rej.max()):.4f}"
    )

# -------------------------
# Accuracy on RewardBench: chosen > max(rejected)
# -------------------------
correct = 0
total = 0

for ex in rb:
    sc_chosen = score_single(ex["chosen"])
    sc_rej = score_batch(ex["rejected"])
    if sc_chosen > sc_rej.max():
        correct += 1
    total += 1

acc = correct / total * 100.0
print(f"\nRewardBench accuracy: {correct}/{total} = {acc:.2f}%")
