"""Shared text-model loading for training and adapter evaluation."""

from __future__ import annotations


def load_base_model(model_name: str, **kwargs):
    from transformers import AutoConfig, AutoModelForCausalLM, Qwen3_5ForCausalLM

    config = AutoConfig.from_pretrained(model_name)
    # Countdown is text-only. The multimodal wrapper retains rope_deltas across generation
    # and scoring batches; those batches need not have equal sizes in GRPO.
    loader = Qwen3_5ForCausalLM if config.model_type == "qwen3_5" else AutoModelForCausalLM
    return loader.from_pretrained(model_name, **kwargs)
