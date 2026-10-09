from pathlib import Path
from typing import Any

import torch


def load_backbone_model(
    model_path: str | Path,
    torch_dtype: torch.dtype,
    backbone_type: str | None = None,
):
    if backbone_type not in {None, "auto", "qwen25vl"}:
        raise ValueError("Only the Qwen2.5-VL backbone is supported")
    from transformers import Qwen2_5_VLForConditionalGeneration

    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )


def backbone_hidden_size(backbone_or_config: Any) -> int:
    config = getattr(backbone_or_config, "config", backbone_or_config)
    value = getattr(config, "hidden_size", None)
    if value is None and getattr(config, "text_config", None) is not None:
        value = getattr(config.text_config, "hidden_size", None)
    if value is None:
        raise ValueError("Could not infer the backbone hidden size")
    return int(value)
