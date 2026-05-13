"""
Plotting utilities.

Generates:
  - Pareto frontier (reward vs KL) per method (DPO, SimPO) sweeping beta.
  - Length distributions per method (KDE over response lengths).
  - Likelihood-displacement curves (chosen/rejected implicit reward over steps).
  - Diversity bar chart (distinct-1, distinct-2 per method).
  - Win-rate vs label-noise line plot.

Reads from `results/` after training runs complete; writes PNGs to
`results/plots/`.

Usage:
  python plotting.py --results-dir ./results --out-dir ./results/plots
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _safe_history(method_dir):
    h = os.path.join(method_dir, "history.json")
    if not os.path.exists(h):
        return None
    return _load_json(h)


def gather_runs(results_dir: str) -> dict:
    """Index method runs by method/config from a results dir.

    Expects layout `<results_dir>/<method>/<config_name>/<timestamp>/` or
    `<results_dir>/<method>/<config_name>/`.
    """
    runs = {}
    for method in ("dpo", "simpo", "ppo", "sft"):
        m_dir = os.path.join(results_dir, method)
        if not os.path.isdir(m_dir):
            continue
        for cfg in sorted(os.listdir(m_dir)):
            cfg_dir = os.path.join(m_dir, cfg)
            if not os.path.isdir(cfg_dir):
                continue
            # Each cfg may have one ts subdir or be flat.
            candidates = [cfg_dir]
            for sub in sorted(os.listdir(cfg_dir)):
                ts_dir = os.path.join(cfg_dir, sub)
                if os.path.isdir(ts_dir):
                    candidates.append(ts_dir)
            for c in candidates:
                history = _safe_history(c)
                config = None
                cfg_path = os.path.join(c, "config.json")
                if os.path.exists(cfg_path):
                    config = _load_json(cfg_path)
                if history is None:
                    continue
                key = f"{method}/{cfg}"
                runs[key] = {"path": c, "method": method, "cfg": cfg,
                             "history": history, "config": config}
    return runs


def plot_pareto(runs: dict, out_path: str):
    """Plot final implicit reward vs KL proxy across beta sweeps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"dpo": "tab:blue", "simpo": "tab:orange", "ppo": "tab:green"}
    for key, run in runs.items():
        method = run["method"]
        if method not in {"dpo", "simpo"}:
            continue
        config = run["config"] or {}
        beta = (config.get("args") or {}).get("beta", None)
        h = run["history"]
        if not h: continue
        # Use mean final-window r_chosen - r_rejected as "reward" proxy
        last = h[-min(5, len(h)):]
        gap = np.mean([row["r_chosen"] - row["r_rejected"] for row in last])
        # KL proxy: drift of chosen reward away from 0 (likelihood displacement)
        drift = abs(np.mean([row["r_chosen"] for row in last]))
        ax.scatter(drift, gap, color=colors[method], s=70,
                   label=f"{method.upper()} (β={beta})" if beta else method.upper())
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=8)
    ax.set_xlabel("|mean r_chosen| (KL proxy)")
    ax.set_ylabel("mean reward gap (r_chosen − r_rejected)")
    ax.set_title("Reward–KL frontier across β")
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"[plot] {out_path}")


def plot_length_curves(runs: dict, out_path: str):
    """Per-method response-length over training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"dpo": "tab:blue", "simpo": "tab:orange", "ppo": "tab:green"}
    for key, run in runs.items():
        method = run["method"]
        h = run["history"]
        if method not in {"dpo", "simpo"} or not h: continue
        steps = [r["step"] for r in h]
        len_w = [r.get("len_chosen", np.nan) for r in h]
        ax.plot(steps, len_w, color=colors[method], label=f"{method.upper()} chosen", alpha=0.8)
    ax.set_xlabel("step"); ax.set_ylabel("avg chosen response length (tokens)")
    ax.set_title("Chosen response length across training")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"[plot] {out_path}")


def plot_likelihood_displacement(runs: dict, out_path: str):
    """Implicit rewards (chosen and rejected) over training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = [k for k, v in runs.items() if v["method"] in {"dpo", "simpo"}]
    n = len(methods)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for i, key in enumerate(methods):
        run = runs[key]
        h = run["history"]
        steps = [r["step"] for r in h]
        rc = [r["r_chosen"] for r in h]
        rr = [r["r_rejected"] for r in h]
        ax = axes[0, i]
        ax.plot(steps, rc, label="r_chosen", color="tab:blue")
        ax.plot(steps, rr, label="r_rejected", color="tab:orange")
        ax.set_xlabel("step"); ax.set_ylabel("implicit reward")
        ax.set_title(key)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"[plot] {out_path}")


def plot_diversity_from_summary(summary_csv: str, out_path: str):
    """Plot distinct-1 / distinct-2 / self-BLEU bars from eval summary."""
    if not os.path.exists(summary_csv):
        print(f"[plot] skipping diversity (no {summary_csv})")
        return
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    with open(summary_csv) as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return
    names = [r["name"] for r in rows]
    d1 = [float(r["distinct_1_method"]) for r in rows]
    d2 = [float(r["distinct_2_method"]) for r in rows]
    sb = [float(r["self_bleu_4_method"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(names, d1); axes[0].set_title("distinct-1"); axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[1].bar(names, d2); axes[1].set_title("distinct-2"); axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[2].bar(names, sb); axes[2].set_title("self-BLEU-4 (lower = more diverse)"); axes[2].set_xticklabels(names, rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"[plot] {out_path}")


def plot_noise_robustness(summary_csv: str, out_path: str):
    """Plot LC win rate vs (method, noise) if rows are named like dpo_noise_0.10."""
    if not os.path.exists(summary_csv):
        return
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    with open(summary_csv) as f:
        r = csv.DictReader(f)
        rows = list(r)
    by_method = {}
    for row in rows:
        name = row["name"]
        try:
            wr = float(row["lc_win_rate"])
        except Exception:
            continue
        if not name.startswith(("dpo_noise_", "simpo_noise_")):
            # Also catch the no-noise baseline rows
            if name in {"dpo_default", "simpo_default"}:
                method = name.split("_")[0]
                by_method.setdefault(method, []).append((0.0, wr))
            continue
        method, _, noise = name.split("_", 2)
        by_method.setdefault(method, []).append((float(noise), wr))
    if not by_method:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"dpo": "tab:blue", "simpo": "tab:orange"}
    for method, pts in by_method.items():
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-o", color=colors.get(method, None), label=method.upper())
    ax.set_xlabel("label-noise probability")
    ax.set_ylabel("LC win rate vs SFT")
    ax.set_title("Robustness to preference-label noise")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"[plot] {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--out-dir", default="./results/plots")
    p.add_argument("--eval-summary", default="./results/eval/summary.csv")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = gather_runs(args.results_dir)
    print(f"[plot] found {len(runs)} runs in {args.results_dir}")

    plot_pareto(runs, os.path.join(args.out_dir, "pareto_reward_kl.png"))
    plot_length_curves(runs, os.path.join(args.out_dir, "length_during_training.png"))
    plot_likelihood_displacement(runs, os.path.join(args.out_dir, "likelihood_displacement.png"))
    plot_diversity_from_summary(args.eval_summary, os.path.join(args.out_dir, "diversity_bars.png"))
    plot_noise_robustness(args.eval_summary, os.path.join(args.out_dir, "noise_robustness.png"))
    print("[plot] done")


if __name__ == "__main__":
    main()
