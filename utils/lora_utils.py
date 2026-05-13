"""LoRA helpers for the Pythia-1B-deduped base model."""

from typing import Optional


_KNOWN_TARGETS = [
    ("query_key_value",),    # Pythia / GPT-NeoX
    ("c_attn",),             # GPT-2 family (incl. sshleifer/tiny-gpt2)
    ("q_proj", "v_proj"),    # LLaMA / Mistral / Qwen
]


def _autodetect_target_modules(model) -> list:
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    for cand in _KNOWN_TARGETS:
        if all(c in names for c in cand):
            return list(cand)
    raise RuntimeError(
        f"add_lora: could not auto-detect target_modules; module names seen: "
        f"{sorted(names)[:20]}..."
    )


def add_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05,
             target_modules: Optional[list] = None):
    """Wrap a HuggingFace causal LM in LoRA adapters via the `peft` library.

    Returns the wrapped model. If `peft` is unavailable we raise a clear error
    rather than silently training the full model.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        raise ImportError(
            "peft is required for LoRA training: `pip install peft accelerate`"
        ) from e

    if target_modules is None:
        # Pythia uses fused QKV under `query_key_value`; tiny-gpt2 uses `c_attn`.
        # Auto-detect so the same helper works for the smoke-test model.
        target_modules = _autodetect_target_modules(model)

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, cfg)


def trainable_param_count(model) -> tuple:
    """Return (trainable_params, total_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
