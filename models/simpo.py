"""
SimPO loss (Meng et al., 2024).

Differences vs DPO:
  - No reference policy. The implicit reward uses only the trainable policy.
  - The implicit reward is **length-normalised** average log-probability over
    response tokens.
  - A target reward margin gamma is subtracted inside the sigmoid (this is
    what enforces a minimum preference gap on every pair).

Implicit reward:
    r_hat(x, y) = (beta / |y|) * sum_t log pi_theta(y_t | x, y_{<t})

Loss:
    L_SimPO = - E[ log sigma( r_hat(y_w) - r_hat(y_l) - gamma ) ]
"""

import torch
import torch.nn.functional as F


def get_avg_logp_response(model, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor,
                          response_mask: torch.Tensor):
    """Return (sum_logp, count, avg_logp) over the response tokens only.

    Args:
        model: causal LM whose forward returns logits as `out.logits`.
        input_ids: (B, T) token ids (prompt + response).
        attention_mask: (B, T) — 1 on real tokens, 0 on padding.
        response_mask: (B, T) — 1 on response tokens, 0 on prompt/padding.

    Returns:
        sum_logp: (B,) summed log p(y_t | x, y_<t) over response tokens.
        n_tok:    (B,) number of response tokens per example (>= 1).
        avg_logp: (B,) sum_logp / n_tok.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # (B, T, V)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = response_mask[:, 1:].contiguous().float()

    logp_per_tok = F.log_softmax(shift_logits, dim=-1)
    logp_label = logp_per_tok.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    logp_label = logp_label * shift_mask

    sum_logp = logp_label.sum(dim=-1)
    n_tok = shift_mask.sum(dim=-1).clamp(min=1.0)
    avg_logp = sum_logp / n_tok
    return sum_logp, n_tok, avg_logp


def simpo_loss(
    policy_avg_logp_w: torch.Tensor,
    policy_avg_logp_l: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 1.0,
):
    """SimPO loss (Meng et al., 2024).

    Args:
        policy_avg_logp_w: (B,) per-example average log-prob of the chosen
            response under the trainable policy (already length-normalised).
        policy_avg_logp_l: (B,) same, for rejected.
        beta: temperature scaling.
        gamma: target reward margin.

    Returns:
        loss:          scalar
        chosen_reward: (scalar) mean implicit reward for chosen, = beta * avg_logp_w
        rejected_reward: (scalar) mean implicit reward for rejected
        accuracy:      (scalar) fraction with chosen reward > rejected reward
        margin:        (scalar) mean gap (chosen_reward - rejected_reward)
    """
    chosen_reward = beta * policy_avg_logp_w
    rejected_reward = beta * policy_avg_logp_l
    logits = (chosen_reward - rejected_reward) - gamma
    loss = -F.logsigmoid(logits).mean()

    accuracy = (chosen_reward > rejected_reward).float().mean()
    margin = (chosen_reward - rejected_reward).mean()
    return (
        loss,
        chosen_reward.mean().detach(),
        rejected_reward.mean().detach(),
        accuracy.detach(),
        margin.detach(),
    )
