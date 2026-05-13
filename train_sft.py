"""
Supervised fine-tuning (SFT) of Pythia-1B-deduped + LoRA on the TL;DR
summarisation prompt/response data. Produces the shared SFT checkpoint that
all three preference-optimisation methods (PPO, DPO, SimPO) initialise from.

Usage:
  python train_sft.py --output-dir ./results/sft
  python train_sft.py --output-dir <azure_output> --azure
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


def _ensure_packages():
    for pkg in [("transformers", "transformers>=4.40"),
                ("datasets", "datasets"),
                ("peft", "peft>=0.10"),
                ("accelerate", "accelerate")]:
        try:
            __import__(pkg[0])
        except ImportError:
            print(f"[deps] installing {pkg[1]}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg[1]])


def main():
    p = argparse.ArgumentParser(description="SFT for Simplification Tax project")
    p.add_argument("--output-dir", type=str, default="./results/sft")
    p.add_argument("--azure", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-name", type=str, default="EleutherAI/pythia-1b-deduped")
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--n-examples", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--bf16", action="store_true", help="enable bf16 training on A100")
    args = p.parse_args()

    _ensure_packages()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.data import load_tldr_sft, SFTDataset, sft_collate
    from utils.lora_utils import add_lora, trainable_param_count

    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[setup] device={device}")
    output_dir = str(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    azure_run = None
    if args.azure:
        from azureml.core import Run
        azure_run = Run.get_context()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if (args.bf16 and device == "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)
    model = add_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    trainable, total = trainable_param_count(model)
    print(f"[model] {args.model_name}: total={total/1e6:.1f}M trainable={trainable/1e6:.2f}M ({100*trainable/total:.2f}%)")

    examples = load_tldr_sft(n=args.n_examples)
    ds = SFTDataset(examples, tokenizer, max_len=args.max_len)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: sft_collate(b, pad_id), num_workers=0)

    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    history = []
    step = 0
    model.train()
    for epoch in range(args.epochs):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            rmask = batch["response_mask"].to(device)
            out = model(input_ids=ids, attention_mask=attn)
            sl = out.logits[:, :-1, :].contiguous().float()
            slabels = ids[:, 1:].contiguous()
            smask = rmask[:, 1:].contiguous().float()
            loss_per_tok = F.cross_entropy(sl.view(-1, sl.size(-1)), slabels.view(-1),
                                           reduction="none").view(slabels.shape)
            loss = (loss_per_tok * smask).sum() / smask.sum().clamp(min=1.0)

            (loss / args.gradient_accumulation_steps).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % args.log_interval == 0:
                print(f"[sft] step={step:5d} epoch={epoch} loss={loss.item():.4f}")
                history.append({"step": step, "epoch": epoch, "loss": float(loss.item())})
                if azure_run is not None:
                    azure_run.log("sft_loss", float(loss.item()))
                    azure_run.log("sft_step", step)
            step += 1

    save_dir = os.path.join(output_dir, "model")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "trainable_params": trainable, "total_params": total,
                   "finished_at": datetime.utcnow().isoformat() + "Z"}, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [h["step"] for h in history]
        plt.figure(figsize=(6, 4))
        plt.plot(steps, [h["loss"] for h in history])
        plt.xlabel("step"); plt.ylabel("SFT loss"); plt.title("SFT training")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sft_loss.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[plot] failed: {e}")

    print("[done] SFT")


if __name__ == "__main__":
    main()
