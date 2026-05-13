"""
DPO training: continue from the SFT checkpoint with frozen reference.

Loss: standard Rafailov et al. (2023) DPO with implicit reward
    r_hat(y|x) = beta * log(pi_theta(y|x) / pi_ref(y|x))

Usage:
  python train_dpo.py --sft-checkpoint ./results/sft/<ts>/model \
                      --output-dir ./results/dpo --beta 0.1
"""
import argparse
import copy
import json
import os
import random
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
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
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg[1]])


def main():
    p = argparse.ArgumentParser(description="DPO for Simplification Tax project")
    p.add_argument("--output-dir", type=str, default="./results/dpo")
    p.add_argument("--azure", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sft-checkpoint", type=str, required=False, default=None,
                   help="path to the LoRA-wrapped SFT checkpoint (model/ folder). "
                        "If omitted, falls back to base Pythia-1B-deduped + fresh LoRA.")
    p.add_argument("--base-model", type=str, default="EleutherAI/pythia-1b-deduped")
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--n-examples", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--label-noise", type=float, default=0.0,
                   help="probability of flipping chosen/rejected per pair")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    args = p.parse_args()

    _ensure_packages()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.data import (load_tldr_preferences, PreferenceDataset,
                            pref_collate, inject_label_noise)
    from utils.lora_utils import add_lora, trainable_param_count
    from models.dpo import get_logp_response, dpo_loss

    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[setup] device={device} beta={args.beta} noise={args.label_noise}")
    output_dir = str(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    azure_run = None
    if args.azure:
        from azureml.core import Run
        azure_run = Run.get_context()

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint or args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if (args.bf16 and device == "cuda") else torch.float32

    def _load_policy():
        base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype).to(device)
        if args.sft_checkpoint and PeftModel is not None:
            print(f"[setup] loading LoRA adapters from {args.sft_checkpoint}")
            return PeftModel.from_pretrained(base, args.sft_checkpoint, is_trainable=True).to(device)
        return add_lora(base, r=args.lora_r, alpha=args.lora_alpha)

    policy = _load_policy()
    ref = _load_policy()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()
    trainable, total = trainable_param_count(policy)
    print(f"[model] policy total={total/1e6:.1f}M trainable={trainable/1e6:.2f}M")

    examples = load_tldr_preferences(n=args.n_examples)
    examples = inject_label_noise(examples, args.label_noise, seed=seed)
    ds = PreferenceDataset(examples, tokenizer, max_len=args.max_len)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: pref_collate(b, pad_id), num_workers=0)

    opt = optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                      lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    history = []
    step = 0
    policy.train()
    for epoch in range(args.epochs):
        for batch in loader:
            ids_w = batch["ids_w"].to(device); attn_w = batch["attn_w"].to(device); rmask_w = batch["rmask_w"].to(device)
            ids_l = batch["ids_l"].to(device); attn_l = batch["attn_l"].to(device); rmask_l = batch["rmask_l"].to(device)
            logp_w = get_logp_response(policy, ids_w, attn_w, rmask_w)
            logp_l = get_logp_response(policy, ids_l, attn_l, rmask_l)
            with torch.no_grad():
                ref_logp_w = get_logp_response(ref, ids_w, attn_w, rmask_w)
                ref_logp_l = get_logp_response(ref, ids_l, attn_l, rmask_l)
            loss, ch_r, rj_r, acc = dpo_loss(logp_w, logp_l, ref_logp_w, ref_logp_l, beta=args.beta)

            (loss / args.gradient_accumulation_steps).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % args.log_interval == 0:
                len_w = float(rmask_w.sum(-1).float().mean().item())
                len_l = float(rmask_l.sum(-1).float().mean().item())
                print(f"[dpo] step={step:5d} loss={loss.item():.4f} acc={acc.item():.3f} "
                      f"r_w={ch_r.item():+.3f} r_l={rj_r.item():+.3f} "
                      f"len_w={len_w:.1f} len_l={len_l:.1f}")
                row = {"step": step, "loss": float(loss.item()),
                       "acc": float(acc.item()),
                       "r_chosen": float(ch_r.item()),
                       "r_rejected": float(rj_r.item()),
                       "len_chosen": len_w, "len_rejected": len_l}
                history.append(row)
                if azure_run is not None:
                    for k, v in row.items():
                        azure_run.log(f"dpo_{k}", v)
            step += 1

    save_dir = os.path.join(output_dir, "model")
    policy.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "trainable_params": trainable, "total_params": total,
                   "finished_at": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    print("[done] DPO")


if __name__ == "__main__":
    main()
