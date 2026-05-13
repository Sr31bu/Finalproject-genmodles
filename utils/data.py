"""
Data loaders for the Simplification Tax project.

Datasets:
  - SFT: `trl-lib/tldr` (TL;DR summarisation prompt + reference summary)
  - Preferences (for DPO / SimPO): `CarperAI/openai_summarize_comparisons`
  - Falls back to in-memory sample data if HuggingFace is unreachable.

All loaders return dicts with consistent keys:
  - SFT example:        {"prompt": str, "response": str}
  - Preference example: {"prompt": str, "chosen": str, "rejected": str}

Helpers also include:
  - response-only tokenisation with prompt-masked loss masks
  - label-noise injection (flip preferences with probability p)
"""

import random
from typing import Iterable, List, Tuple

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# HuggingFace loaders with offline fallback
# ---------------------------------------------------------------------------

def _normalize_tldr_sft_row(row) -> dict:
    """Normalise a row from trl-lib/tldr -> {prompt, response}."""
    prompt = row.get("prompt") or row.get("post") or row.get("content")
    response = row.get("completion") or row.get("label") or row.get("summary")
    if prompt is None or response is None:
        return None
    prompt = str(prompt).strip()
    response = str(response).strip()
    if not prompt.endswith("\nTL;DR:") and "TL;DR" not in prompt:
        prompt = prompt + "\nTL;DR:"
    return {"prompt": prompt + " ", "response": response}


def _normalize_pref_row(row) -> dict:
    """Normalise a row from CarperAI/openai_summarize_comparisons."""
    prompt = row.get("prompt") or row.get("post")
    chosen = row.get("chosen") or row.get("response_chosen")
    rejected = row.get("rejected") or row.get("response_rejected")
    if prompt is None or chosen is None or rejected is None:
        return None
    prompt = str(prompt).strip()
    if "TL;DR" not in prompt:
        prompt = prompt + "\nTL;DR:"
    return {
        "prompt": prompt + " ",
        "chosen": str(chosen).strip(),
        "rejected": str(rejected).strip(),
    }


def load_tldr_sft(n: int = 4096, split: str = "train"):
    """Load TL;DR SFT examples; fall back to builtin samples if unreachable."""
    try:
        from datasets import load_dataset
        ds = load_dataset("trl-lib/tldr", split=f"{split}[:{n}]")
        out = [_normalize_tldr_sft_row(r) for r in ds]
        out = [x for x in out if x is not None]
        if out:
            print(f"[data] loaded {len(out)} SFT examples from trl-lib/tldr")
            return out
        raise RuntimeError("empty SFT split")
    except Exception as e:
        print(f"[data] SFT HF load failed: {e}. Falling back to builtin samples.")
        return _builtin_sft(n)


def load_tldr_preferences(n: int = 4096, split: str = "train"):
    """Load preference pairs; fall back to builtin samples if unreachable."""
    try:
        from datasets import load_dataset
        ds = load_dataset("CarperAI/openai_summarize_comparisons", split=f"{split}[:{n}]")
        out = [_normalize_pref_row(r) for r in ds]
        out = [x for x in out if x is not None]
        if out:
            print(f"[data] loaded {len(out)} preference pairs from CarperAI/openai_summarize_comparisons")
            return out
        raise RuntimeError("empty pref split")
    except Exception as e:
        print(f"[data] preference HF load failed: {e}. Falling back to builtin samples.")
        return _builtin_prefs(n)


# ---------------------------------------------------------------------------
# Builtin fallback samples (TL;DR-style)
# ---------------------------------------------------------------------------

_SAMPLE_POSTS = [
    "I bought a used car last week and the engine started making a weird sound after two days. The dealer says it's normal but my friend who's a mechanic says it could be the timing belt. I don't know whether to push for a refund or get a second opinion.",
    "My roommate keeps leaving the kitchen a mess. I've talked to her three times now and she promises to clean but never does. Should I just buy a label maker and assign chores or have a serious conversation?",
    "I got into both my top-choice college and a state school with a full scholarship. My parents want me to take the scholarship but I feel like I'd regret not going to the dream school. I don't know what to do.",
    "I started a new job a month ago and my manager keeps taking credit for my work in front of the team. I've documented everything but I'm worried about being seen as difficult if I bring it up.",
]
_SAMPLE_GOOD = [
    "Used car making weird sound; dealer dismissive, mechanic friend suspects timing belt — should I refund or get second opinion?",
    "Messy roommate ignores cleanup chats — labels and chores or another talk?",
    "Choice between scholarship and dream college — how to decide?",
    "New manager taking credit for my work — speak up or stay quiet?",
]
_SAMPLE_BAD = [
    "I have a car issue, what should I do, please help me decide between many options.",
    "Roommate is bad, asking what to do.",
    "Picking a college, hard.",
    "Manager problem at work.",
]


def _builtin_sft(n: int):
    out = []
    for i in range(n):
        post = _SAMPLE_POSTS[i % len(_SAMPLE_POSTS)]
        good = _SAMPLE_GOOD[i % len(_SAMPLE_GOOD)]
        out.append({"prompt": post + "\nTL;DR: ", "response": good})
    return out


def _builtin_prefs(n: int):
    out = []
    for i in range(n):
        idx = i % len(_SAMPLE_POSTS)
        out.append({
            "prompt": _SAMPLE_POSTS[idx] + "\nTL;DR: ",
            "chosen": _SAMPLE_GOOD[idx],
            "rejected": _SAMPLE_BAD[idx],
        })
    return out


# ---------------------------------------------------------------------------
# Label noise
# ---------------------------------------------------------------------------

def inject_label_noise(pref_examples: List[dict], flip_prob: float, seed: int = 0):
    """Flip preferred/rejected with probability `flip_prob`."""
    if flip_prob <= 0:
        return pref_examples
    rng = random.Random(seed)
    out = []
    flipped = 0
    for ex in pref_examples:
        if rng.random() < flip_prob:
            out.append({"prompt": ex["prompt"],
                        "chosen": ex["rejected"],
                        "rejected": ex["chosen"]})
            flipped += 1
        else:
            out.append(dict(ex))
    print(f"[noise] flipped {flipped}/{len(pref_examples)} labels (p={flip_prob})")
    return out


# ---------------------------------------------------------------------------
# Torch Datasets
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """Builds (input_ids, attention_mask, response_mask) per example."""

    def __init__(self, examples, tokenizer, max_len: int = 512):
        self.examples, self.tokenizer, self.max_len = examples, tokenizer, max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = ex["prompt"]
        response = ex["response"]
        p_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        eos = self.tokenizer.eos_token or ""
        r_ids = self.tokenizer.encode(response + eos, add_special_tokens=False)
        ids = (p_ids + r_ids)[: self.max_len]
        # response_mask = 1 on response tokens, 0 on prompt
        mask = [0] * min(len(p_ids), len(ids)) + [1] * max(0, len(ids) - len(p_ids))
        mask = mask[: self.max_len]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "rmask": torch.tensor(mask, dtype=torch.long),
        }


def sft_collate(batch, pad_id: int):
    L = max(len(b["ids"]) for b in batch); B = len(batch)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    rmask = torch.zeros((B, L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["ids"])
        ids[i, :n] = b["ids"]
        attn[i, :n] = 1
        rmask[i, :n] = b["rmask"]
    return {"input_ids": ids, "attention_mask": attn, "response_mask": rmask}


class PreferenceDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len: int = 512):
        self.examples, self.tokenizer, self.max_len = examples, tokenizer, max_len

    def __len__(self):
        return len(self.examples)

    def _encode(self, prompt, response):
        p_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        eos = self.tokenizer.eos_token or ""
        r_ids = self.tokenizer.encode(response + eos, add_special_tokens=False)
        ids = (p_ids + r_ids)[: self.max_len]
        mask = [0] * min(len(p_ids), len(ids)) + [1] * max(0, len(ids) - len(p_ids))
        mask = mask[: self.max_len]
        return ids, mask

    def __getitem__(self, idx):
        ex = self.examples[idx]
        ids_w, m_w = self._encode(ex["prompt"], ex["chosen"])
        ids_l, m_l = self._encode(ex["prompt"], ex["rejected"])
        return {
            "ids_w": torch.tensor(ids_w, dtype=torch.long),
            "mask_w": torch.tensor(m_w, dtype=torch.long),
            "ids_l": torch.tensor(ids_l, dtype=torch.long),
            "mask_l": torch.tensor(m_l, dtype=torch.long),
        }


def pref_collate(batch, pad_id: int):
    def pad(ids_list, mask_list):
        L = max(len(s) for s in ids_list); B = len(ids_list)
        ids = torch.full((B, L), pad_id, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        rmask = torch.zeros((B, L), dtype=torch.long)
        for i, (s, m) in enumerate(zip(ids_list, mask_list)):
            n = len(s)
            ids[i, :n] = s
            attn[i, :n] = 1
            rmask[i, :n] = m
        return ids, attn, rmask

    ids_w, attn_w, rmask_w = pad([b["ids_w"] for b in batch], [b["mask_w"] for b in batch])
    ids_l, attn_l, rmask_l = pad([b["ids_l"] for b in batch], [b["mask_l"] for b in batch])
    return {
        "ids_w": ids_w, "attn_w": attn_w, "rmask_w": rmask_w,
        "ids_l": ids_l, "attn_l": attn_l, "rmask_l": rmask_l,
    }
