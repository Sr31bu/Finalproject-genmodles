# The Simplification Tax — Code

Three-way comparison of **PPO / DPO / SimPO** on TL;DR summarisation with Pythia-1B-deduped + LoRA. All three preference-optimisation losses are implemented from scratch (PPO uses TRL because of its complexity).

## Layout

```
finalproj_genmodels/
├── models/
│   ├── dpo.py            # Rafailov-style DPO loss + get_logp_response (copied from PS4)
│   ├── simpo.py          # SimPO loss: length-normalised reward, no reference
│   └── eval_judge.py     # GPT-4o pairwise judge, LC win rate, distinct-n, self-BLEU
├── utils/
│   ├── data.py           # TL;DR + preference loaders (with offline fallback) + noise injection
│   └── lora_utils.py     # peft LoRA wrapper
├── train_sft.py          # SFT (Pythia-1B-deduped + LoRA r=16, α=32) on trl-lib/tldr
├── train_dpo.py          # DPO continuation from SFT checkpoint
├── train_simpo.py        # SimPO continuation from SFT checkpoint
├── train_ppo.py          # PPO via TRL on top of SFT (merged LoRA + fresh LoRA + value head)
├── eval_runner.py        # Pairwise evaluation against the SFT baseline
├── plotting.py           # Pareto, length, displacement, diversity, noise plots
├── final_project.ipynb   # Single-GPU end-to-end Colab/A100-runnable notebook

```

## Quickstart (single GPU — Colab A100 / local A100)

```bash
# 1. SFT
python train_sft.py --output-dir ./results/sft/run1 --bf16
# 2. DPO/SimPO/PPO (any order; all read the same SFT checkpoint)
python train_dpo.py    --sft-checkpoint ./results/sft/run1/model --output-dir ./results/dpo/default   --beta 0.1 --bf16
python train_simpo.py  --sft-checkpoint ./results/sft/run1/model --output-dir ./results/simpo/default --beta 2.0 --gamma 1.0 --bf16
python train_ppo.py    --sft-checkpoint ./results/sft/run1/model --output-dir ./results/ppo/default   --kl-beta 0.05 --bf16
# 3. Evaluate (uses GPT-4o; set OPENAI_API_KEY first)
python eval_runner.py --checkpoint ./results/dpo/default/model   --sft-checkpoint ./results/sft/run1/model --eval-name dpo_default   --output-dir ./results/eval --n-eval 500 --n-judge 200 --bf16
python eval_runner.py --checkpoint ./results/simpo/default/model --sft-checkpoint ./results/sft/run1/model --eval-name simpo_default --output-dir ./results/eval --n-eval 500 --n-judge 200 --bf16
python eval_runner.py --checkpoint ./results/ppo/default/model   --sft-checkpoint ./results/sft/run1/model --eval-name ppo_default   --output-dir ./results/eval --n-eval 500 --n-judge 200 --bf16
# 4. Plot
python plotting.py --results-dir ./results --out-dir ./results/plots --eval-summary ./results/eval/summary.csv
```

Or just open `final_project.ipynb` and run all.

## GPU Training

All runs were executed on a single A100 GPU. Run `final_project.ipynb` end-to-end or use the individual training scripts as shown in the Quickstart above.

## Configurations covered by `final_project.ipynb`
- 1 × SFT
- 1 × default (DPO β=0.1, SimPO β=2.0/γ=1.0, PPO KL=0.05)
- β sweep: DPO {0.01, 0.05, 0.5, 1.0}, SimPO {0.5, 1.0, 5.0}
- γ sweep: SimPO {0.5, 1.5, 2.0}
- Noise ablation: DPO + SimPO each at {0.10, 0.25, 0.50}

Total: ~17 jobs. Wall-clock ~10 h with reasonable cluster availability.

## Loss equations (paper-aligned)

**DPO** (Rafailov et al., 2023)
```
L_DPO = - E [ log σ ( β [ log π_θ(y_w|x)/π_ref(y_w|x)  -  log π_θ(y_l|x)/π_ref(y_l|x) ] ) ]
```

**SimPO** (Meng et al., 2024)
```
r̂(y|x)   = (β / |y|) · Σ_t log π_θ(y_t | x, y_<t)        # length-normalised
L_SimPO  = - E [ log σ ( r̂(y_w) - r̂(y_l) - γ ) ]
```

**PPO (RLHF)**
```
max_θ E[ r_φ(x, y) - β_KL · KL(π_θ || π_ref) ]
```
with PPO clipped policy gradient (handled by TRL).

## Metrics
- **Length-controlled win rate** — logistic regression of P(method wins) on `Δ_len = len(method) − len(SFT)`. Reports predicted win rate at `Δ_len = 0` so length doesn't bias the comparison.
- **distinct-1 / distinct-2** — fraction of unique uni/bi-grams across all generations.
- **self-BLEU-4** — average BLEU of every generation against the rest; lower = more diverse.
- **Implicit-reward trajectories** — per-step `r_chosen`/`r_rejected` from DPO and SimPO; the gap is what's directly optimised, the absolute drift surfaces likelihood displacement.
- **Wall-clock + peak GPU memory** — automatically logged by A100 ML / `torch.cuda.max_memory_allocated`.

## Requirements

The training scripts auto-install missing deps (`transformers ≥4.40`, `peft ≥0.10`, `datasets`, `accelerate`, `trl ≥0.7.10`, `openai ≥1.0`) at the top of `main()`. If your A100 env (`the training environment`) already has most of these, the install is a no-op.

## Notes

- `models/dpo.py` is the identical module shipped with PSet 4. Reused intentionally.
- HuggingFace `Anthropic/hh-rlhf` is the proposal's secondary dataset; this code targets the proposal's primary dataset (`CarperAI/openai_summarize_comparisons` for preferences, `trl-lib/tldr` for SFT). Offline fallback kicks in only if HF is unreachable.
- The PPO reward model defaults to `cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr` with a fallback to `OpenAssistant/reward-model-deberta-v3-large-v2` if cleanrl's HF entry is missing/private.
