# Output Data Specification

**Purpose**: this document describes the exact shape, schema, and expected value ranges of every artifact the training and evaluation pipeline produces.

All numeric ranges are anchored to the closest published behaviour at Pythia-1B scale on TL;DR summarisation (Rafailov et al. 2023 — DPO; Meng et al. 2024 — SimPO; Xu et al. 2024 — PPO-vs-DPO; Stiennon et al. 2020 — TL;DR reward models). Where literature is silent at this exact scale, ranges are interpolated and flagged.

Scope:
- 17 training runs (see §1)
- 16 pairwise evaluations against the SFT baseline
- ≈970 training history rows, ≈16,000 generated completions, ≈6,400 GPT-4o judgments

---

## 1. Run inventory

Every directory below corresponds to one independent training job.

| Method | Config name | n_runs | β     | γ    | label_noise | Reference |
|--------|-------------|:-:|:-:|:-:|:-:|---|
| SFT    | (only)      | 1 | —     | —    | —           | — |
| DPO    | default     | 1 | 0.10  | —    | 0.00        | SFT |
| DPO    | beta_0.01   | 1 | 0.01  | —    | 0.00        | SFT |
| DPO    | beta_0.05   | 1 | 0.05  | —    | 0.00        | SFT |
| DPO    | beta_0.5    | 1 | 0.50  | —    | 0.00        | SFT |
| DPO    | beta_1.0    | 1 | 1.00  | —    | 0.00        | SFT |
| DPO    | noise_0.10  | 1 | 0.10  | —    | 0.10        | SFT |
| DPO    | noise_0.25  | 1 | 0.10  | —    | 0.25        | SFT |
| DPO    | noise_0.50  | 1 | 0.10  | —    | 0.50        | SFT |
| SimPO  | default     | 1 | 2.00  | 1.00 | 0.00        | none |
| SimPO  | beta_0.5    | 1 | 0.50  | 1.00 | 0.00        | none |
| SimPO  | beta_1.0    | 1 | 1.00  | 1.00 | 0.00        | none |
| SimPO  | beta_5.0    | 1 | 5.00  | 1.00 | 0.00        | none |
| SimPO  | gamma_0.5   | 1 | 2.00  | 0.50 | 0.00        | none |
| SimPO  | gamma_1.5   | 1 | 2.00  | 1.50 | 0.00        | none |
| SimPO  | gamma_2.0   | 1 | 2.00  | 2.00 | 0.00        | none |
| SimPO  | noise_0.10  | 1 | 2.00  | 1.00 | 0.10        | none |
| SimPO  | noise_0.25  | 1 | 2.00  | 1.00 | 0.25        | none |
| SimPO  | noise_0.50  | 1 | 2.00  | 1.00 | 0.50        | none |
| PPO    | default     | 1 | KL=0.05 | —  | —           | SFT |

Total: **20 directories** (1 SFT + 19 preference jobs), (1 SFT + 19 preference jobs).

---

## 2. Directory tree

```
results/
├── sft/<timestamp>/
│   ├── model/                       # peft LoRA adapter (≈30 MB)
│   ├── history.json
│   ├── config.json
│   └── sft_loss.png
├── dpo/<config_name>/<timestamp>/
│   ├── model/                       # peft LoRA adapter
│   ├── history.json
│   └── config.json
├── simpo/<config_name>/<timestamp>/
│   └── … (same shape as dpo)
├── ppo/<config_name>/<timestamp>/
│   └── … (same shape, plus TRL value head)
├── eval/
│   ├── summary.csv                  # flat aggregate, 1 row per evaluated method
│   ├── <method>_<config>.json       # full per-prompt eval
│   └── …
└── plots/
    ├── pareto_reward_kl.png
    ├── length_during_training.png
    ├── likelihood_displacement.png
    ├── diversity_bars.png
    └── noise_robustness.png
```

`<timestamp>` is `YYYYMMDD_HHMMSS` UTC.

---

## 3. `history.json` — training history

A JSON list, one dict per logged step. Logging interval is **20 steps** for DPO/SimPO, **20 steps** for SFT, **10 PPO batches** for PPO.

### 3.1 DPO history

Schema (per row):
```json
{
  "step":         int,
  "loss":         float,
  "acc":          float,
  "r_chosen":     float,
  "r_rejected":   float,
  "len_chosen":   float,
  "len_rejected": float
}
```

Field-by-field:

| Field | Type | Units | Typical range | Notes |
|---|---|---|---|---|
| `step` | int | step | 0, 20, 40, …, 1020 | 51 rows total at batch_size=4, n_examples=4096 |
| `loss` | float | nats | start ≈ 0.69 → end ≈ 0.40–0.60 | starts near log 2 (uniform-pref baseline), drops as policy widens gap |
| `acc` | float | — | start 0.50, end 0.55–0.75 (avg over run) | quantised to {0, 0.25, 0.5, 0.75, 1.0} with B=4 |
| `r_chosen` | float | nats (β·log-ratio) | start 0.0, drifts to −1.5 to −3.5 by end | both rewards drift DOWN — likelihood displacement |
| `r_rejected` | float | nats | start 0.0, drifts to −2.0 to −4.0 | drifts further than r_chosen |
| `len_chosen` | float | tokens | 35 – 70, drifts UP over training | DPO length exploitation |
| `len_rejected` | float | tokens | 30 – 50 | rejected responses do NOT lengthen as much |

Mechanism behind drift: DPO maximises `r_chosen − r_rejected`. Empirically both probabilities can decrease in absolute terms while the gap stays positive — this is the published "likelihood displacement" phenomenon (Razin et al. 2024).

### 3.2 SimPO history

Same as DPO + one extra column:

```json
{
  ..., "margin": float
}
```

| Field | Typical range | Notes |
|---|---|---|
| `loss` | start ≈ 0.69 → end ≈ 0.30–0.50 | margin γ makes the early loss slightly lower |
| `r_chosen` | β·avg_logp ≈ −1.5 to −4.0 at β=2 | absolute value (not log-ratio); larger β = larger magnitude |
| `r_rejected` | slightly more negative than r_chosen | gap forces this |
| `margin` | starts near 0, ends 1.0 – 2.5 | needs to exceed γ for the loss to saturate |
| `len_chosen` | 35 – 50 (less drift than DPO) | length normalisation removes the bloat incentive |

### 3.3 PPO history

```json
{
  "step":         int,
  "mean_reward":  float,
  "kl":           float,
  "loss_total":   float,
  "loss_policy":  float,
  "loss_value":   float
}
```

| Field | Typical range |
|---|---|
| `mean_reward` | starts near reward-model baseline (≈0 standardised), drifts to +0.5 – +2.5 by end |
| `kl` | starts ≈ 0.01, grows to 4 – 8 by end (controlled by `init_kl_coef=0.05`) |
| `loss_total` | starts ≈ 1.0, ends 0.2 – 0.5 |
| `loss_policy` | similar magnitude to total |
| `loss_value` | usually larger than policy loss early, decays |

### 3.4 SFT history

```json
{ "step": int, "epoch": int, "loss": float }
```

| Field | Typical range |
|---|---|
| `loss` | start ≈ 6.5 (cross-entropy on Pythia next-token over response), end ≈ 2.0–2.8 |

---

## 4. `config.json`

```json
{
  "args": {
    "<all CLI args>": <value>
  },
  "trainable_params": int,
  "total_params": int,
  "finished_at": "2026-05-12T08:30:42.123456Z"
}
```

`trainable_params` for Pythia-1B-deduped + LoRA r=16: **≈ 6.3 M** (covers `query_key_value` + `dense` modules). `total_params` ≈ **1.01 B**.

---

## 5. `eval/<method>_<config>.json` — full pairwise evaluation

```json
{
  "name":          str,
  "checkpoint":    str,
  "sft_baseline":  str,
  "n_eval":        int,
  "n_judged":      int,
  "judge":         "gpt-4o",
  "win_rate":      { … },
  "diversity":     { … },
  "lengths":       { … },
  "method_generations": [ … ],
  "sft_generations":    [ … ],
  "judgments":          [ … ],
  "finished_at":   ISO8601 str
}
```

### 5.1 `win_rate` sub-object

| Field | Range | Notes |
|---|---|---|
| `raw_win_rate` | 0 – 1 | (#A wins + 0.5·ties) / n_judged |
| `lc_win_rate` | 0 – 1 | logistic regression of win on length difference, predicted at Δlen=0 |
| `alpha` | usually small | length coefficient; large positive = judge prefers longer A |
| `mean_len_a` | float | tokens, method response |
| `mean_len_b` | float | tokens, SFT baseline |

Mean values by method:

| Config | raw_win_rate | lc_win_rate | alpha | mean_len_a | mean_len_b |
|---|---|---|---|---|---|
| `ppo_default`      | 0.58 – 0.68 | 0.55 – 0.65 | small  | 28 – 40 | 28 – 40 |
| `dpo_default`      | 0.55 – 0.68 | 0.50 – 0.60 | +0.02–+0.05 | 55 – 80 | 28 – 40 |
| `simpo_default`    | 0.55 – 0.65 | 0.55 – 0.65 | small  | 35 – 48 | 28 – 40 |
| `dpo_noise_0.10`   | 0.50 – 0.62 | 0.48 – 0.58 | +0.02 | 50 – 75 | 28 – 40 |
| `dpo_noise_0.25`   | 0.45 – 0.55 | 0.43 – 0.53 | +0.02 | 50 – 75 | 28 – 40 |
| `dpo_noise_0.50`   | 0.38 – 0.50 | 0.35 – 0.48 | +0.02 | 50 – 75 | 28 – 40 |
| `simpo_noise_0.10` | 0.52 – 0.63 | 0.52 – 0.63 | small | 35 – 48 | 28 – 40 |
| `simpo_noise_0.25` | 0.48 – 0.58 | 0.48 – 0.58 | small | 35 – 48 | 28 – 40 |
| `simpo_noise_0.50` | 0.45 – 0.55 | 0.45 – 0.55 | small | 35 – 48 | 28 – 40 |
| `dpo_beta_0.01`    | 0.50 – 0.60 | 0.50 – 0.60 | small | small drift from SFT (β too small to move policy) |
| `dpo_beta_0.05`    | 0.54 – 0.65 | 0.50 – 0.60 | +0.02 | drift toward DPO default |
| `dpo_beta_0.5`     | 0.55 – 0.68 | 0.50 – 0.62 | +0.02 | stronger length drift |
| `dpo_beta_1.0`     | 0.55 – 0.70 | 0.45 – 0.60 | +0.03 | very long outputs, judge fooled |
| `simpo_beta_0.5`   | 0.52 – 0.60 | 0.52 – 0.60 | small | similar to default |
| `simpo_beta_1.0`   | 0.55 – 0.62 | 0.55 – 0.62 | small | |
| `simpo_beta_5.0`   | 0.48 – 0.60 | 0.48 – 0.60 | small | high β too aggressive, may oscillate |
| `simpo_gamma_0.5`  | 0.55 – 0.65 | 0.55 – 0.65 | small | weaker margin |
| `simpo_gamma_1.5`  | 0.55 – 0.65 | 0.55 – 0.65 | small | |
| `simpo_gamma_2.0`  | 0.55 – 0.65 | 0.55 – 0.65 | small | strong margin, sharper preferences |

### 5.2 `diversity` sub-object

| Field | Range | Notes |
|---|---|---|
| `method_distinct_1` | 0.10 – 0.30 | fraction unique unigrams (lower = more repetitive) |
| `method_distinct_2` | 0.30 – 0.60 | bigrams; more discriminative than 1-grams |
| `method_self_bleu_4` | 0.20 – 0.65 | LOWER = more diverse (less self-overlap) |
| `sft_distinct_1`   | 0.18 – 0.28 | reference baseline; SFT is generally more diverse than DPO |
| `sft_distinct_2`   | 0.45 – 0.62 | |
| `sft_self_bleu_4`  | 0.20 – 0.40 | |

Predicted method ordering on diversity (best → worst): **PPO > SimPO > DPO**. PPO retains diversity due to on-policy sampling; DPO collapses toward modal responses.

### 5.3 `lengths` sub-object

| Field | Notes |
|---|---|
| `method_mean_len`   | DPO ≈ 2× SFT; SimPO ≈ 1.1× SFT; PPO ≈ 1.0× SFT |
| `method_median_len` | similar shape, less affected by tail |
| `sft_mean_len`      | 28 – 40 tokens |
| `sft_median_len`    | 26 – 38 tokens |

### 5.4 `method_generations`, `sft_generations` arrays

500 entries each, schema:
```json
{
  "prompt":     str,
  "completion": str,
  "length":     int   // token count of completion only
}
```

### 5.5 `judgments` array

200 entries (with `both_orderings=True`, the harness invokes GPT-4o twice per pair):

```json
{
  "idx":         int,
  "winner":      "A" | "B" | "tie",
  "fwd":         "A" | "B" | "tie",
  "rev":         "A" | "B" | "tie",
  "raw_forward": str,
  "raw_reverse": str,
  "a_len":       int,
  "b_len":       int
}
```

Expected distribution of `winner` (rounded, debiased):
- Strong method (e.g. PPO): **A ≈ 55–65 %, B ≈ 25–35 %, tie ≈ 10–15 %**
- Mid method (DPO, SimPO): **A ≈ 50–60 %, B ≈ 30–40 %, tie ≈ 10–15 %**
- Tie rate climbs sharply when the two methods are close in quality.

`fwd` vs `rev` disagreement rate is typically **10–20 %** with GPT-4o-2024-08-06 on TL;DR; that's why both-orderings debiasing matters.

---

## 6. `eval/summary.csv` — flat aggregate

One row per evaluated checkpoint. **Expected: 16 rows × 14 columns** (PPO + 8 DPO + 7 SimPO; the SFT baseline is not evaluated against itself).

Columns (in order):
```
name                  str
checkpoint            str
lc_win_rate           float ∈ [0, 1]
raw_win_rate          float ∈ [0, 1]
alpha_len             float                   # length coefficient
mean_len_method       float                   # tokens
mean_len_sft          float
distinct_1_method     float ∈ [0, 1]
distinct_2_method     float ∈ [0, 1]
self_bleu_4_method    float ∈ [0, 1]          # lower = more diverse
distinct_1_sft        float
distinct_2_sft        float
self_bleu_4_sft       float
n_judged              int                     # = 200 in default config
```

---

## 7. Cross-config patterns (use these for sanity checks)

| Pattern | Expected behaviour |
|---|---|
| β sweep (DPO) | `lc_win_rate` peaks at β≈0.05–0.1, drops at β≥0.5 (over-regularised), drops at β=0.01 (under-regularised) |
| β sweep (SimPO) | `lc_win_rate` peaks at β≈2.0, flat shape; small β = weak signal, large β = unstable |
| γ sweep (SimPO) | monotonic improvement up to γ≈1.5, then plateau or slight degradation |
| label_noise | DPO degrades **faster** than SimPO (steeper slope). 50 % noise reduces DPO `lc_win_rate` by 0.15–0.20; SimPO by 0.08–0.12 |
| length over training (DPO) | `len_chosen` monotonically increases by 15–30 tokens over the run |
| length over training (SimPO) | `len_chosen` roughly flat (drift < 5 tokens) |
| accuracy over training | both DPO and SimPO climb from 0.5 to 0.65–0.80 average by end |
| chosen-vs-rejected gap | strictly positive by step ~200 in every healthy run |

---

## 8. Red-flag values (something is broken)

| Symptom | Diagnosis |
|---|---|
| `loss` is NaN at any step | gradient blow-up; check learning rate, gradient clipping |
| `acc` stays at exactly 0.5 for >200 steps | reference and policy identical, training not happening |
| `r_chosen ≈ r_rejected` throughout | β too small; preferences not being learned |
| `len_chosen` blows past 200 tokens | length exploitation runaway — almost only DPO at high β |
| `lc_win_rate < 0.4` for the default DPO/SimPO | data loading failed |
| `kl` (PPO) growing unbounded > 30 | KL coefficient too low, policy diverged from reference |
| All `judgments` show `winner == "A"` | order bias; check `both_orderings` flag |
| `alpha_len > 0.05` | judge has substantial length bias; LC adjustment is doing real work |

---

## 9. Realistic full pipeline summary CSV (illustrative, for shape only)

This is what `eval/summary.csv` looks like after running the full pipeline.

```
name,                checkpoint,     lc_win_rate, raw_win_rate, alpha_len, mean_len_method, mean_len_sft, distinct_1_method, distinct_2_method, self_bleu_4_method, distinct_1_sft, distinct_2_sft, self_bleu_4_sft, n_judged
ppo_default,         <path>,         0.62,        0.63,         0.005,     34.0,            34.0,         0.24,              0.52,              0.34,               0.23,           0.52,           0.36,            200
dpo_default,         <path>,         0.55,        0.62,         0.030,     65.0,            34.0,         0.18,              0.42,              0.52,               0.23,           0.52,           0.36,            200
simpo_default,       <path>,         0.60,        0.60,         0.008,     41.0,            34.0,         0.21,              0.47,              0.42,               0.23,           0.52,           0.36,            200
dpo_beta_0.01,       <path>,         0.55,        0.55,         0.010,     38.0,            34.0,         0.22,              0.50,              0.40,               0.23,           0.52,           0.36,            200
dpo_beta_0.05,       <path>,         0.57,        0.61,         0.020,     58.0,            34.0,         0.19,              0.44,              0.48,               0.23,           0.52,           0.36,            200
dpo_beta_0.5,        <path>,         0.55,        0.65,         0.035,     78.0,            34.0,         0.17,              0.40,              0.55,               0.23,           0.52,           0.36,            200
dpo_beta_1.0,        <path>,         0.50,        0.65,         0.040,     85.0,            34.0,         0.15,              0.36,              0.60,               0.23,           0.52,           0.36,            200
simpo_beta_0.5,      <path>,         0.56,        0.56,         0.008,     38.0,            34.0,         0.21,              0.47,              0.42,               0.23,           0.52,           0.36,            200
simpo_beta_1.0,      <path>,         0.59,        0.59,         0.008,     40.0,            34.0,         0.21,              0.47,              0.42,               0.23,           0.52,           0.36,            200
simpo_beta_5.0,      <path>,         0.54,        0.54,         0.008,     42.0,            34.0,         0.19,              0.45,              0.46,               0.23,           0.52,           0.36,            200
simpo_gamma_0.5,     <path>,         0.59,        0.59,         0.008,     40.0,            34.0,         0.21,              0.47,              0.42,               0.23,           0.52,           0.36,            200
simpo_gamma_1.5,     <path>,         0.60,        0.60,         0.008,     41.0,            34.0,         0.20,              0.46,              0.43,               0.23,           0.52,           0.36,            200
simpo_gamma_2.0,     <path>,         0.60,        0.60,         0.008,     41.0,            34.0,         0.20,              0.45,              0.44,               0.23,           0.52,           0.36,            200
dpo_noise_0.10,      <path>,         0.53,        0.60,         0.030,     63.0,            34.0,         0.18,              0.42,              0.52,               0.23,           0.52,           0.36,            200
dpo_noise_0.25,      <path>,         0.49,        0.56,         0.030,     63.0,            34.0,         0.17,              0.41,              0.55,               0.23,           0.52,           0.36,            200
dpo_noise_0.50,      <path>,         0.42,        0.50,         0.030,     63.0,            34.0,         0.16,              0.39,              0.58,               0.23,           0.52,           0.36,            200
simpo_noise_0.10,    <path>,         0.58,        0.58,         0.008,     41.0,            34.0,         0.21,              0.47,              0.42,               0.23,           0.52,           0.36,            200
simpo_noise_0.25,    <path>,         0.55,        0.55,         0.008,     41.0,            34.0,         0.20,              0.46,              0.43,               0.23,           0.52,           0.36,            200
simpo_noise_0.50,    <path>,         0.50,        0.50,         0.008,     41.0,            34.0,         0.20,              0.45,              0.44,               0.23,           0.52,           0.36,            200
```

These are the expected midpoints of the ranges in §5.1 and §5.2. Actual measured values will vary due to stochasticity in training and evaluation.
