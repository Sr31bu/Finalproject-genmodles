"""
Evaluation utilities:
  - GPT-4o pairwise judge with both-orderings debiasing
  - Length-controlled win rate (AlpacaEval-style)
  - Diversity metrics: distinct-n, self-BLEU
  - Likelihood-displacement diagnostics

Requires `OPENAI_API_KEY` in the environment for the GPT-4o judge. The
implementation only contacts the OpenAI API when `pairwise_gpt4o_judge` is
called; the diversity / LC-win-rate functions are pure NumPy.
"""

import os
import time
import json
import math
import random
from typing import Callable, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# GPT-4o pairwise judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating summaries of Reddit posts. You will see a post and two candidate summaries (A and B). Choose which summary is better. Consider:
1. Faithfulness — does the summary reflect what's actually in the post?
2. Conciseness — is it appropriately short without losing meaning?
3. Coverage — does it capture the main point?
4. Coherence — is it well-written?

Respond ONLY with a single JSON object: {"winner": "A"} or {"winner": "B"} or {"winner": "tie"}.
Do not include any other text."""


def _call_openai_chat(client, messages, model="gpt-4o-2024-08-06",
                     max_retries=4, base_wait=2.0):
    """Resilient ChatCompletion call with exponential backoff."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=20,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            wait = base_wait * (2 ** attempt)
            print(f"[judge] OpenAI call failed ({e}); retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"OpenAI judge failed after {max_retries} retries: {last_err}")


def _parse_winner(text: str) -> str:
    """Parse model output, return 'A' | 'B' | 'tie'."""
    try:
        obj = json.loads(text)
        w = str(obj.get("winner", "")).strip().lower()
        if w in {"a", "b", "tie"}:
            return w.upper() if w != "tie" else "tie"
    except Exception:
        pass
    # Fall back to substring search
    t = text.lower()
    if '"a"' in t or "winner: a" in t:
        return "A"
    if '"b"' in t or "winner: b" in t:
        return "B"
    return "tie"


def pairwise_gpt4o_judge(
    post: str,
    response_a: str,
    response_b: str,
    *,
    model: str = "gpt-4o-2024-08-06",
    both_orderings: bool = True,
    client=None,
):
    """Compare two responses with GPT-4o. Returns a dict.

    Schema:
        {
          "winner": "A" | "B" | "tie",
          "raw_forward": "...", "raw_reverse": "..." | None,
          "fwd": "A" | "B" | "tie",
          "rev": "A" | "B" | "tie" | None,
        }

    When both_orderings is True we run the judge twice (A first, then B first)
    and report:
      - winner = "A" if (fwd=='A' and rev=='A')
      - winner = "B" if (fwd=='B' and rev=='B')
      - winner = "tie" otherwise (judge disagrees with itself)
    """
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Install openai>=1.0: `pip install openai`") from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        client = OpenAI(api_key=api_key)

    def _ask(a, b):
        user = (
            f"Reddit post:\n```\n{post}\n```\n\n"
            f"Summary A:\n```\n{a}\n```\n\n"
            f"Summary B:\n```\n{b}\n```\n\n"
            "Which summary is better?"
        )
        raw = _call_openai_chat(client, [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ], model=model)
        return raw, _parse_winner(raw)

    raw_fwd, fwd = _ask(response_a, response_b)
    if not both_orderings:
        return {
            "winner": fwd, "fwd": fwd, "rev": None,
            "raw_forward": raw_fwd, "raw_reverse": None,
        }
    raw_rev_in_swap, rev_swap = _ask(response_b, response_a)
    # rev_swap was judged with original B placed first; flip it back
    rev = {"A": "B", "B": "A", "tie": "tie"}[rev_swap]
    if fwd == rev and fwd != "tie":
        winner = fwd
    else:
        winner = "tie"
    return {
        "winner": winner, "fwd": fwd, "rev": rev,
        "raw_forward": raw_fwd, "raw_reverse": raw_rev_in_swap,
    }


# ---------------------------------------------------------------------------
# Win rate / length control
# ---------------------------------------------------------------------------

def raw_win_rate(judgments: List[dict]) -> float:
    """Fraction of decisions where method "A" wins (ties count as 0.5)."""
    n = 0; s = 0.0
    for j in judgments:
        w = j["winner"]
        if w == "A": s += 1.0
        elif w == "tie": s += 0.5
        n += 1
    return s / max(n, 1)


def length_controlled_win_rate(
    judgments: List[dict],
    lens_a: List[int],
    lens_b: List[int],
    alpha: Optional[float] = None,
) -> dict:
    """AlpacaEval-style length-controlled win rate.

    We fit a logistic regression of P(A wins) on the length-difference
    (len(A) - len(B)) and report the predicted win rate at delta_len = 0.

    If alpha is given it is used directly, otherwise alpha is found by
    1-dim Newton's method on the logistic likelihood.

    Returns:
        {
          "raw_win_rate":     float,
          "lc_win_rate":      float,
          "alpha":            float,  # length coefficient (positive => A is preferred for being longer)
          "mean_len_a":       float,
          "mean_len_b":       float,
        }
    """
    assert len(judgments) == len(lens_a) == len(lens_b)
    n = len(judgments)
    if n == 0:
        return {"raw_win_rate": float("nan"), "lc_win_rate": float("nan"),
                "alpha": 0.0, "mean_len_a": 0.0, "mean_len_b": 0.0}

    y = np.array([
        1.0 if j["winner"] == "A" else 0.0 if j["winner"] == "B" else 0.5
        for j in judgments
    ])
    delta = np.array([lens_a[i] - lens_b[i] for i in range(n)], dtype=float)
    # Standardise delta so alpha is unitless
    std = float(delta.std()) if float(delta.std()) > 1e-8 else 1.0
    delta_n = delta / std

    # logistic regression: P(y=1) = sigmoid(beta0 + alpha * delta_n)
    beta0 = 0.0
    a = 0.0 if alpha is None else float(alpha) * std
    if alpha is None:
        for _ in range(50):
            z = beta0 + a * delta_n
            p = 1.0 / (1.0 + np.exp(-z))
            g0 = (p - y).mean()
            ga = ((p - y) * delta_n).mean()
            h00 = (p * (1.0 - p)).mean()
            ha0 = (p * (1.0 - p) * delta_n).mean()
            haa = (p * (1.0 - p) * delta_n * delta_n).mean()
            # 2x2 Newton step
            det = h00 * haa - ha0 * ha0 + 1e-8
            d_beta = (haa * g0 - ha0 * ga) / det
            d_a = (-ha0 * g0 + h00 * ga) / det
            beta0 -= d_beta
            a -= d_a
            if abs(d_beta) + abs(d_a) < 1e-6:
                break

    lc_p = 1.0 / (1.0 + math.exp(-beta0))
    return {
        "raw_win_rate": float(y.mean()),
        "lc_win_rate": float(lc_p),
        "alpha": float(a / std),  # rescale back to per-token
        "mean_len_a": float(np.mean(lens_a)),
        "mean_len_b": float(np.mean(lens_b)),
    }


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(0, len(tokens) - n + 1)]


def distinct_n(samples: Sequence[str], n: int = 1, tokenizer: Optional[Callable] = None):
    """Fraction of unique n-grams across all samples."""
    if tokenizer is None:
        tokenizer = lambda s: s.split()
    all_grams = []
    for s in samples:
        toks = tokenizer(s)
        all_grams.extend(_ngrams(toks, n))
    if not all_grams:
        return 0.0
    return len(set(all_grams)) / len(all_grams)


def self_bleu(samples: Sequence[str], n: int = 4, max_pairs: int = 200,
              tokenizer: Optional[Callable] = None, seed: int = 42):
    """Mean BLEU-n of every sample against the rest (lower = more diverse).

    Sampling up to `max_pairs` random pairs to keep cost bounded.
    """
    if tokenizer is None:
        tokenizer = lambda s: s.split()
    if len(samples) < 2:
        return 0.0
    tokenised = [tokenizer(s) for s in samples]
    rng = random.Random(seed)
    pairs = []
    n_samples = len(samples)
    for _ in range(max_pairs):
        i = rng.randrange(n_samples)
        j = rng.randrange(n_samples - 1)
        if j >= i:
            j += 1
        pairs.append((i, j))
    scores = []
    for i, j in pairs:
        scores.append(_bleu_n(tokenised[i], tokenised[j], n))
    return float(np.mean(scores))


def _bleu_n(hypothesis, reference, n):
    """BLEU-n for a single (hyp, ref) pair, with brevity penalty."""
    if not hypothesis:
        return 0.0
    weights = [1.0 / n] * n
    prec_logs = []
    for k in range(1, n + 1):
        hyp_grams = _ngrams(hypothesis, k)
        ref_grams = _ngrams(reference, k)
        if not hyp_grams:
            return 0.0
        from collections import Counter
        ref_counts = Counter(ref_grams)
        clipped = 0
        for g in hyp_grams:
            if ref_counts[g] > 0:
                clipped += 1
                ref_counts[g] -= 1
        precision = (clipped + 1.0) / (len(hyp_grams) + 1.0)  # smoothing
        prec_logs.append(weights[k - 1] * math.log(precision))
    bp = 1.0 if len(hypothesis) > len(reference) else math.exp(1.0 - len(reference) / max(1, len(hypothesis)))
    return bp * math.exp(sum(prec_logs))


# ---------------------------------------------------------------------------
# Likelihood displacement
# ---------------------------------------------------------------------------

def likelihood_displacement(history: List[dict]) -> dict:
    """From a training history with per-step `r_chosen` and `r_rejected`,
    compute aggregate displacement statistics. Used by both DPO and SimPO
    trainers (note: for SimPO `r_chosen/rejected` are absolute, not log-ratios)."""
    if not history:
        return {}
    rc = np.array([h.get("r_chosen", float("nan")) for h in history], dtype=float)
    rr = np.array([h.get("r_rejected", float("nan")) for h in history], dtype=float)
    gap = rc - rr
    return {
        "final_r_chosen": float(rc[-1]),
        "final_r_rejected": float(rr[-1]),
        "mean_gap": float(np.nanmean(gap)),
        "frac_positive_gap": float(np.nanmean(gap > 0)),
        "max_displacement": float(np.nanmax(np.abs(rc - rc[0]))),
    }
