"""
DPO loss + utilities for PS4.2.

Implements:
  - Sequence log-probability under a causal LM, masked to response tokens only.
  - DPO loss: -log sigma(beta * (log pi_theta(y_w)/pi_ref(y_w) - log pi_theta(y_l)/pi_ref(y_l))).
  - Implicit reward: r_hat(x, y) = beta * log(pi_theta(y|x) / pi_ref(y|x)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_logp_response(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                     response_mask: torch.Tensor) -> torch.Tensor:
    """Compute summed log-probability of the response tokens under model.

    Args:
        model: causal LM (forward returns CausalLMOutput with logits).
        input_ids: (B, T)
        attention_mask: (B, T) — 1 on real tokens, 0 on padding.
        response_mask: (B, T) — 1 on response tokens (where loss should apply), 0 elsewhere.

    Returns:
        logp: (B,) summed log p(y_t | x, y_<t) over response tokens only.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # (B, T, V)

    # Shift: logits[:, :-1] predict input_ids[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = response_mask[:, 1:].contiguous().float()

    # Per-token log prob
    logp_per_tok = F.log_softmax(shift_logits, dim=-1)
    logp_label = logp_per_tok.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    # Mask out non-response tokens (prompt + padding)
    logp_label = logp_label * shift_mask
    logp_seq = logp_label.sum(dim=-1)  # (B,)
    return logp_seq


def dpo_loss(
    pi_theta_logp_w: torch.Tensor,
    pi_theta_logp_l: torch.Tensor,
    pi_ref_logp_w: torch.Tensor,
    pi_ref_logp_l: torch.Tensor,
    beta: float = 0.1,
):
    """DPO loss (Rafailov et al. 2023).

    L = -log sigma( beta * ( (logp_theta(y_w) - logp_ref(y_w)) - (logp_theta(y_l) - logp_ref(y_l)) ) )

    Returns:
        loss: scalar
        chosen_reward: implicit reward for chosen
        rejected_reward: implicit reward for rejected
        accuracy: fraction where chosen_reward > rejected_reward
    """
    chosen_logratio = pi_theta_logp_w - pi_ref_logp_w
    rejected_logratio = pi_theta_logp_l - pi_ref_logp_l
    logits = beta * (chosen_logratio - rejected_logratio)
    loss = -F.logsigmoid(logits).mean()

    chosen_reward = beta * chosen_logratio.detach()
    rejected_reward = beta * rejected_logratio.detach()
    accuracy = (chosen_reward > rejected_reward).float().mean()
    return loss, chosen_reward.mean(), rejected_reward.mean(), accuracy
