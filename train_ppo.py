"""
PPO training via TRL on Pythia-1B-deduped + LoRA.

We use TRL's `PPOTrainer` with a pretrained reward model. The course's
proposal lists the cleanrl reward model checkpoint; we default to a public
HuggingFace reward model trained on the same TL;DR preferences as a fallback
if the cleanrl artifact is not accessible.

Usage:
  python train_ppo.py --sft-checkpoint ./results/sft/<ts>/model \
                      --output-dir ./results/ppo --kl-beta 0.05
"""
import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch


REWARD_MODEL_CHOICES = [
    # cleanrl artifacts may be private or moved; this is the primary attempt
    "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr",
    # public TL;DR reward model widely used in DPO / SimPO ablation papers
    "OpenAssistant/reward-model-deberta-v3-large-v2",
]


def _ensure_packages():
    for pkg in [("transformers", "transformers>=4.40"),
                ("datasets", "datasets"),
                ("peft", "peft>=0.10"),
                ("accelerate", "accelerate"),
                ("trl", "trl>=0.7.10")]:
        try:
            __import__(pkg[0])
        except ImportError:
            print(f"[deps] installing {pkg[1]}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg[1]])


def _try_load_reward(model_name, device):
    """Load a sequence-classification reward model. Fall back across choices."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tried = []
    last_err = None
    options = [model_name] + [m for m in REWARD_MODEL_CHOICES if m != model_name]
    for name in options:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForSequenceClassification.from_pretrained(name).to(device)
            mdl.eval()
            print(f"[reward] loaded {name}")
            return name, mdl, tok
        except Exception as e:
            tried.append((name, str(e)))
            last_err = e
    raise RuntimeError(f"Failed to load any reward model. Tried: {tried}; last_err={last_err}")


def main():
    p = argparse.ArgumentParser(description="PPO for Simplification Tax project (TRL-backed)")
    p.add_argument("--output-dir", type=str, default="./results/ppo")
    p.add_argument("--azure", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sft-checkpoint", type=str, default=None,
                   help="path to LoRA SFT checkpoint (model/ folder)")
    p.add_argument("--base-model", type=str, default="EleutherAI/pythia-1b-deduped")
    p.add_argument("--reward-model", type=str, default=REWARD_MODEL_CHOICES[0])
    p.add_argument("--n-prompts", type=int, default=2048,
                   help="number of prompts to PPO-train over")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--mini-batch-size", type=int, default=2)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.4e-5)
    p.add_argument("--kl-beta", type=float, default=0.05,
                   help="KL penalty coefficient")
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--max-prompt-len", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--log-interval", type=int, default=10)
    args = p.parse_args()

    _ensure_packages()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None
    # TRL imports
    try:
        from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    except ImportError:
        # newer TRL may have moved things around; surface a clear error
        raise

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.data import load_tldr_preferences  # reuse for prompts only
    from utils.lora_utils import add_lora, trainable_param_count

    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[setup] device={device} kl_beta={args.kl_beta}")
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
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype).to(device)
    if args.sft_checkpoint and PeftModel is not None:
        print(f"[setup] loading LoRA adapters from {args.sft_checkpoint}")
        base = PeftModel.from_pretrained(base, args.sft_checkpoint, is_trainable=True).to(device)
        # Merge LoRA into base weights so TRL's value head works on the merged model.
        try:
            base = base.merge_and_unload()
            print("[setup] merged LoRA weights into base")
        except Exception as e:
            print(f"[setup] merge_and_unload failed: {e}; continuing with adapters")

    # Re-attach a fresh LoRA on top of the SFT-merged weights so PPO updates
    # only trainable adapters; this matches the proposal's "LoRA on Pythia-1B".
    policy_with_lora = add_lora(base, r=args.lora_r, alpha=args.lora_alpha)
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(policy_with_lora)
    trainable, total = trainable_param_count(policy)
    print(f"[model] policy total={total/1e6:.1f}M trainable={trainable/1e6:.2f}M")

    # Reward model
    rm_name, reward_model, reward_tok = _try_load_reward(args.reward_model, device)

    # Build dataset of prompts (use preference data's prompt fields)
    examples = load_tldr_preferences(n=args.n_prompts)
    prompts = [e["prompt"] for e in examples]

    def tokenize_prompts(prompts):
        return tokenizer(prompts, padding=False, truncation=True,
                         max_length=args.max_prompt_len, return_tensors=None)["input_ids"]

    tokenised_prompts = tokenize_prompts(prompts)

    # PPO config
    config = PPOConfig(
        model_name=args.base_model,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        init_kl_coef=args.kl_beta,
        target_kl=6.0,
        cliprange=args.clip_range,
        cliprange_value=args.clip_range,
        seed=seed,
    )

    ppo = PPOTrainer(config=config, model=policy, ref_model=None,
                     tokenizer=tokenizer)

    def reward_fn(prompts_text, responses_text):
        """Compute scalar reward from the loaded reward model."""
        rewards = []
        bs = 8
        with torch.no_grad():
            for i in range(0, len(prompts_text), bs):
                chunk_p = prompts_text[i:i + bs]
                chunk_r = responses_text[i:i + bs]
                # Many TL;DR reward models expect concatenated prompt+response.
                inputs = reward_tok([p + r for p, r in zip(chunk_p, chunk_r)],
                                    padding=True, truncation=True, max_length=512,
                                    return_tensors="pt").to(device)
                logits = reward_model(**inputs).logits.squeeze(-1)  # (B,)
                rewards.extend(logits.float().tolist())
        return [torch.tensor(r, dtype=torch.float32) for r in rewards]

    history = []
    step = 0
    n_total = len(tokenised_prompts)
    bs = args.batch_size
    policy.train()
    for start in range(0, n_total, bs):
        batch_ids = tokenised_prompts[start:start + bs]
        if len(batch_ids) < bs:
            break
        query_tensors = [torch.tensor(ids, dtype=torch.long).to(device) for ids in batch_ids]
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "top_k": 0,
            "top_p": 1.0,
            "temperature": 1.0,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        response_tensors = ppo.generate(query_tensors, return_prompt=False, **gen_kwargs)
        prompts_text = [tokenizer.decode(q, skip_special_tokens=True) for q in query_tensors]
        responses_text = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
        rewards = reward_fn(prompts_text, responses_text)

        stats = ppo.step(query_tensors, response_tensors, rewards)

        if step % args.log_interval == 0:
            mean_r = float(np.mean([r.item() for r in rewards]))
            kl = float(stats.get("objective/kl", 0.0))
            print(f"[ppo] step={step:5d} mean_reward={mean_r:+.3f} kl={kl:.4f}")
            row = {"step": step, "mean_reward": mean_r, "kl": kl,
                   "loss_total": float(stats.get("ppo/loss/total", 0.0)),
                   "loss_policy": float(stats.get("ppo/loss/policy", 0.0)),
                   "loss_value": float(stats.get("ppo/loss/value", 0.0))}
            history.append(row)
            if azure_run is not None:
                for k, v in row.items():
                    azure_run.log(f"ppo_{k}", v)
        step += 1

    save_dir = os.path.join(output_dir, "model")
    policy.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args),
                   "reward_model_used": rm_name,
                   "trainable_params": trainable, "total_params": total,
                   "finished_at": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    print("[done] PPO")


if __name__ == "__main__":
    main()
