"""Model configurations and mappings."""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    hf_model_path: str
    is_chat_model: bool = False
    description: str = ""


# Predefined model configurations
MODEL_CONFIGS = {
    "qwen": ModelConfig(
        name="qwen",
        hf_model_path="Qwen/Qwen2.5-7B",
        is_chat_model=False,
        description="Qwen 2.5 7B base model"
    ),
    "qwen_math": ModelConfig(
        name="qwen_math",
        hf_model_path="Qwen/Qwen2.5-Math-7B",
        is_chat_model=False,
        description="Qwen 2.5 Math 7B specialized for mathematics"
    ),
    "qwen_math_grpo": ModelConfig(
        name="qwen_math_grpo",
        hf_model_path="stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150",
        is_chat_model=True,
        description="Qwen Math 7B fine-tuned with GRPO"
    ),
    "phi": ModelConfig(
        name="phi",
        hf_model_path="microsoft/Phi-3.5-mini-instruct",
        is_chat_model=True,
        description="Microsoft Phi-3.5 Mini Instruct"
    ),
    "phi_grpo": ModelConfig(
        name="phi_grpo",
        hf_model_path="",  # Add actual path when available
        is_chat_model=True,
        description="Phi-3.5 fine-tuned with GRPO"
    ),
    "tulu": ModelConfig(
        name="tulu",
        hf_model_path="allenai/Llama-3.1-Tulu-3-8B-DPO",
        is_chat_model=True,
        description="Llama 3.1 Tulu 3 8B with DPO"
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]


def list_available_models() -> Dict[str, str]:
    """List all available models with descriptions."""
    return {name: config.description for name, config in MODEL_CONFIGS.items()}


def add_custom_model(name: str, hf_model_path: str, is_chat_model: bool = False, description: str = ""):
    """Add a custom model configuration."""
    MODEL_CONFIGS[name] = ModelConfig(
        name=name,
        hf_model_path=hf_model_path,
        is_chat_model=is_chat_model,
        description=description
    )