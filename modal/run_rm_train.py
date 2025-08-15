import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import modal

app : modal.App = modal.App("rm-train")
image = (
    modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel")
    .apt_install("build-essential", "libaio-dev", "ninja-build", "git", "ffmpeg")
    # Ensure CUDA env vars are visible & nvcc on PATH
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
    })
    # (Usually not needed, but guarantees `python` exists where Modal expects it)
    .run_commands("ln -sf $(which python) /usr/bin/python").pip_install(
    "trl[vllm]",
    "accelerate",
    "deepspeed",
    "datasets",
    "transformers",
    "trl",
    "torch",
    "seaborn",
    "matplotlib",
    "wandb",
    "tf-keras", 
    "pandas",
    "numpy",
    "scikit-learn")
)


MODELS_DIR = Path("/models")
vol: modal.Volume = modal.Volume.from_name("RLHF")


import os
import modal

N_GPUS   = int(os.environ.get("N_GPUS", "8"))          # how many GPUs to use
GPU_TYPE = os.environ.get("GPU_TYPE", "H100")          # or "H100"
GPU_SPEC = f"{GPU_TYPE}:{N_GPUS}" if N_GPUS > 1 else GPU_TYPE

@app.function(
    image=image,
    gpu=GPU_SPEC,                                      # e.g. "A100:4"
    timeout=60 * 60 * 3,
    volumes={"/workspace": vol},
    secrets=[modal.Secret.from_name("huggingface-secret"),
             modal.Secret.from_name("wandb-secret")]
)
def train_reward():
    import os, sys, subprocess, shutil

    os.chdir("/workspace")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf_cache")
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

    # sanity: clamp N_GPUS to what's actually visible
    try:
        import torch
        avail = max(torch.cuda.device_count(), 1)
    except Exception:
        avail = 1
    use_gpus = min(N_GPUS, avail)
    if use_gpus != N_GPUS:
        print(f"[warn] Requested {N_GPUS} GPUs, but only {avail} visible. Using {use_gpus}.")
    N = use_gpus

    env = os.environ.copy()
    # NCCL tips for single-node, multi-GPU
    env.setdefault("NCCL_DEBUG", "INFO")
    env.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    env.setdefault("NCCL_IB_DISABLE", "1")   # no IB on Modal; avoids hangs
    
    # Let Accelerate assign RANK/LOCAL_RANK to subprocesses.
    # Parent just provides rendezvous for safety:
    if N > 1:
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env.setdefault("MASTER_PORT", "29500")
    else:
        # single-GPU defaults so scripts reading these don't crash
        env.setdefault("RANK", "0")
        env.setdefault("LOCAL_RANK", "0")
        env.setdefault("WORLD_SIZE", "1")
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env.setdefault("MASTER_PORT", "29500")

    # optional diagnostics
    try:
        print("CUDA devices:", avail)
        output = subprocess.check_call(["nvidia-smi"], text=True)
        print(output)
        if shutil.which("nvcc"):
            subprocess.check_call(["nvcc", "--version"])
        else:
            print("[info] nvcc not on PATH (fine unless using DeepSpeed build).")
    except Exception as e:
        print("[diag] gpu check failed:", e)

    # build accelerate command
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", str(N),
        "--num_machines", "1",
        "train_llama.py",
        "--per_device_train_batch_size", "16",
        "--per_device_eval_batch_size", "16",
        "--gradient_accumulation_steps", "2",
        "--max_length", "2048",
        "--gradient_checkpointing",
        "--num_train_epochs", "1",
        "--learning_rate", "2e-6",
        "--deepspeed", "deepspeed_config.json",
        "--output_path", "./models/llama3_8b_rm_fair"
    ]

    print("Launching:", " ".join(cmd))
    subprocess.check_call(cmd, env=env)
@app.local_entrypoint()
def main(): 
    train_reward.remote()