"""
Evaluation runner.

For each trained checkpoint:
  1. Generate N completions on the held-out TL;DR eval split.
  2. Compute diversity (distinct-1, distinct-2, self-BLEU).
  3. Run GPT-4o pairwise judge against the SFT baseline with both-orderings
     debiasing -> length-controlled win rate.
  4. Write per-row judge results to results/eval/<checkpoint_name>.json and
     aggregate metrics into results/eval/summary.csv.

Usage:
  python eval_runner.py --checkpoint ./results/dpo/2026.../model \
                         --sft-checkpoint ./results/sft/2026.../model \
                         --output-dir ./results/eval/dpo_default \
                         --n-eval 500 --judge gpt-4o
  python eval_runner.py --batch-eval ./results/eval_config.json
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from typing import List

import numpy as np
import torch


def _ensure_packages():
    for pkg in [("transformers", "transformers>=4.40"),
                ("datasets", "datasets"),
                ("peft", "peft>=0.10"),
                ("openai", "openai>=1.0")]:
        try:
            __import__(pkg[0])
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg[1]])


def load_policy(checkpoint_path: str, base_model: str, device, dtype):
    """Load either a LoRA checkpoint (has adapter_config.json) or a full model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype).to(device)
        model = PeftModel.from_pretrained(base, checkpoint_path).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=dtype).to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_completions(model, tokenizer, prompts, device, max_new_tokens=64,
                         temperature=0.7, top_p=0.95, batch_size=8):
    """Generate one completion per prompt."""
    outs = []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_id,
        )
        # Strip prompt from each output
        for j in range(out.shape[0]):
            prompt_len = int(enc["attention_mask"][j].sum().item())
            full_ids = out[j].tolist()
            response_ids = full_ids[prompt_len:]
            text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            outs.append({"prompt": batch[j], "completion": text,
                         "length": len(tokenizer.encode(text, add_special_tokens=False))})
    return outs


def evaluate_one(args):
    """Run the full evaluation pipeline for one (method, sft-baseline) pair."""
    _ensure_packages()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.data import load_tldr_preferences
    from models.eval_judge import (pairwise_gpt4o_judge, raw_win_rate,
                                   length_controlled_win_rate, distinct_n,
                                   self_bleu)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.bf16 and device == "cuda") else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    # Eval prompts (hold-out slice — last n examples of the preference data)
    full_examples = load_tldr_preferences(n=args.n_eval * 2, split="train")
    eval_prompts = [e["prompt"] for e in full_examples[-args.n_eval:]]
    posts = [p.split("\nTL;DR:")[0] for p in eval_prompts]

    # Method model
    print(f"[eval] loading method checkpoint: {args.checkpoint}")
    method_model, method_tok = load_policy(args.checkpoint, args.base_model, device, dtype)
    method_outs = generate_completions(method_model, method_tok, eval_prompts, device,
                                       max_new_tokens=args.max_new_tokens,
                                       temperature=args.gen_temperature,
                                       top_p=args.gen_top_p,
                                       batch_size=args.gen_batch_size)
    del method_model
    if device == "cuda": torch.cuda.empty_cache()

    # SFT baseline
    print(f"[eval] loading SFT checkpoint: {args.sft_checkpoint}")
    sft_model, sft_tok = load_policy(args.sft_checkpoint, args.base_model, device, dtype)
    sft_outs = generate_completions(sft_model, sft_tok, eval_prompts, device,
                                    max_new_tokens=args.max_new_tokens,
                                    temperature=args.gen_temperature,
                                    top_p=args.gen_top_p,
                                    batch_size=args.gen_batch_size)
    del sft_model
    if device == "cuda": torch.cuda.empty_cache()

    # Diversity
    method_texts = [o["completion"] for o in method_outs]
    sft_texts = [o["completion"] for o in sft_outs]
    diversity = {
        "method_distinct_1": distinct_n(method_texts, 1),
        "method_distinct_2": distinct_n(method_texts, 2),
        "method_self_bleu_4": self_bleu(method_texts, 4),
        "sft_distinct_1": distinct_n(sft_texts, 1),
        "sft_distinct_2": distinct_n(sft_texts, 2),
        "sft_self_bleu_4": self_bleu(sft_texts, 4),
    }
    lengths = {
        "method_mean_len": float(np.mean([o["length"] for o in method_outs])),
        "method_median_len": float(np.median([o["length"] for o in method_outs])),
        "sft_mean_len": float(np.mean([o["length"] for o in sft_outs])),
        "sft_median_len": float(np.median([o["length"] for o in sft_outs])),
    }

    # Judge
    judgments = []
    lens_a, lens_b = [], []
    if args.judge != "none":
        client = None
        if args.judge.startswith("gpt-4o") or args.judge == "openai":
            from openai import OpenAI
            if "OPENAI_API_KEY" not in os.environ:
                raise RuntimeError("OPENAI_API_KEY not set in environment.")
            client = OpenAI()
        n_judged = min(args.n_judge, len(method_outs))
        print(f"[judge] running pairwise judge over {n_judged} prompts ({args.judge})")
        for i in range(n_judged):
            post = posts[i]
            a = method_outs[i]["completion"]
            b = sft_outs[i]["completion"]
            j = pairwise_gpt4o_judge(post, a, b, model="gpt-4o-2024-08-06",
                                     both_orderings=args.both_orderings,
                                     client=client)
            j["idx"] = i
            j["a_len"] = method_outs[i]["length"]
            j["b_len"] = sft_outs[i]["length"]
            judgments.append(j)
            lens_a.append(method_outs[i]["length"])
            lens_b.append(sft_outs[i]["length"])
            if (i + 1) % 25 == 0:
                print(f"[judge] {i+1}/{n_judged}")
        wr = length_controlled_win_rate(judgments, lens_a, lens_b)
    else:
        wr = {"raw_win_rate": float("nan"), "lc_win_rate": float("nan"),
              "alpha": 0.0,
              "mean_len_a": lengths["method_mean_len"],
              "mean_len_b": lengths["sft_mean_len"]}

    # Persist
    name = args.eval_name or os.path.basename(args.checkpoint.rstrip("/"))
    out = {
        "name": name,
        "checkpoint": args.checkpoint,
        "sft_baseline": args.sft_checkpoint,
        "n_eval": len(method_outs),
        "n_judged": len(judgments),
        "judge": args.judge,
        "win_rate": wr,
        "diversity": diversity,
        "lengths": lengths,
        "method_generations": method_outs,
        "sft_generations": sft_outs,
        "judgments": judgments,
        "finished_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(args.output_dir, f"{name}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] wrote {os.path.join(args.output_dir, name+'.json')}")

    # Append/update summary.csv
    summary_path = os.path.join(args.output_dir, "summary.csv")
    row = {
        "name": name,
        "checkpoint": args.checkpoint,
        "lc_win_rate": wr["lc_win_rate"],
        "raw_win_rate": wr["raw_win_rate"],
        "alpha_len": wr["alpha"],
        "mean_len_method": lengths["method_mean_len"],
        "mean_len_sft": lengths["sft_mean_len"],
        "distinct_1_method": diversity["method_distinct_1"],
        "distinct_2_method": diversity["method_distinct_2"],
        "self_bleu_4_method": diversity["method_self_bleu_4"],
        "distinct_1_sft": diversity["sft_distinct_1"],
        "distinct_2_sft": diversity["sft_distinct_2"],
        "self_bleu_4_sft": diversity["sft_self_bleu_4"],
        "n_judged": len(judgments),
    }
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[eval] appended row to {summary_path}")


def main():
    p = argparse.ArgumentParser(description="Pairwise evaluator for Simplification Tax")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sft-checkpoint", required=True)
    p.add_argument("--output-dir", default="./results/eval")
    p.add_argument("--eval-name", default=None,
                   help="output file name; defaults to basename(checkpoint)")
    p.add_argument("--base-model", default="EleutherAI/pythia-1b-deduped")
    p.add_argument("--n-eval", type=int, default=500,
                   help="number of held-out prompts to generate from")
    p.add_argument("--n-judge", type=int, default=200,
                   help="how many of those to send to GPT-4o judge")
    p.add_argument("--judge", default="gpt-4o",
                   choices=["gpt-4o", "openai", "none"],
                   help="set to 'none' to skip the API call and only compute diversity/length")
    p.add_argument("--both-orderings", action="store_true", default=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--gen-temperature", type=float, default=0.7)
    p.add_argument("--gen-top-p", type=float, default=0.95)
    p.add_argument("--gen-batch-size", type=int, default=8)
    p.add_argument("--bf16", action="store_true")
    args = p.parse_args()

    evaluate_one(args)


if __name__ == "__main__":
    main()
