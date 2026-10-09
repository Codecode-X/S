#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import jsonlines
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlamaForCausalLM, Qwen2_5_VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ssvr.reasoning import (
    ACTION_ID_TO_NAME_LOWER,
    action_names_for_meta,
    build_qwen_visual_start_tokens_index,
    is_action_legal,
    is_maze_flip_meta,
    is_minibehaviour_meta,
    load_local_action_model,
    normalize_state,
    qwen_visual_image_from_tokens,
    qwen_visual_prompt_text,
    qwen_visual_tokens_for_sample,
    maze_flip_view_action_to_world,
    state_carrying,
    state_to_task_position,
    task_distance_to_goal,
    transition_step,
)
from ssvr.evaluation.metrics import compute_em_pr


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def apply_action_mask_to_logits(
    logits: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if action_mask is None:
        return logits
    mask = action_mask.to(device=logits.device, dtype=torch.float32)
    if mask.shape != logits.shape:
        raise ValueError(
            f"action_mask shape {tuple(mask.shape)} must match logits shape {tuple(logits.shape)}"
        )
    if torch.any(mask.sum(dim=-1) <= 0):
        raise ValueError("action_mask must leave at least one action unmasked per sample")
    return logits.masked_fill(mask <= 0.0, -1e9)


def minibehaviour_interaction_action_mask(
    meta: Mapping[str, Any], state: int
) -> Tuple[float, ...]:
    action_names = action_names_for_meta(meta)
    if not is_minibehaviour_meta(meta):
        return tuple(1.0 for _ in action_names)
    mask = [1.0 for _ in action_names]
    name_to_id = {str(name).upper(): idx for idx, name in enumerate(action_names)}
    for name in ("PICK", "DROP"):
        action_id = name_to_id[name]
        mask[action_id] = 1.0 if is_action_legal(meta, state, action_id) else 0.0
    return tuple(mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--backbone_type", choices=("qwen25vl", "lvm"), default="qwen25vl")
    parser.add_argument("--processor_path", default=None)
    parser.add_argument("--test_dataset", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--markdown_path", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--image_scale", type=float, default=1.0)
    parser.add_argument("--torch_dtype", choices=sorted(DTYPE_MAP), default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--attribution_mode",
        choices=("grad_x_attention", "raw_attention"),
        default="grad_x_attention",
        help=(
            "grad_x_attention is class-specific for the predicted action logit. "
            "raw_attention only shows where the appended state token attends."
        ),
    )
    parser.add_argument("--attn_layer", default="all", help="'last', 'all', or a zero-based layer index.")
    parser.add_argument("--attn_head", default="mean", help="'mean' or a zero-based head index.")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_scan_samples", type=int, default=None)
    parser.add_argument("--cases_per_bucket", type=int, default=1)
    parser.add_argument(
        "--min_correct_move_steps",
        type=int,
        default=0,
        help=(
            "Only select correct cases with at least this many navigation "
            "actions. Useful for MiniBehaviour, where early correct cases may "
            "be PICK->DROP interaction-only samples."
        ),
    )
    parser.add_argument("--num_sample_shards", type=int, default=1)
    parser.add_argument("--sample_shard_index", type=int, default=0)
    parser.add_argument("--levels", default=None, help="Comma-separated levels. Defaults to all levels in dataset.")
    parser.add_argument(
        "--select_only",
        action="store_true",
        help="Only scan and write selected_cases.json/scan_selected_cases.csv; skip heatmap rendering.",
    )
    parser.add_argument(
        "--rescan_cases",
        action="store_true",
        help="Ignore an existing selected_cases.json and rescan correct/wrong cases.",
    )
    parser.add_argument(
        "--minibehaviour_interaction_action_masking",
        action="store_true",
        help=(
            "Apply the same MiniBehaviour interaction mask used by closed-loop "
            "evaluation: PICK is only available before carrying at printer "
            "neighbors; DROP is only available while carrying at table neighbors."
        ),
    )
    parser.add_argument(
        "--minibehaviour_split_attribution",
        action="store_true",
        help=(
            "For MiniBehaviour, additionally render separate attribution maps "
            "for navigation, PICK, and DROP logits at each rollout step."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--reproduce_command",
        default="bash scripts/attention/visualize_8gpu.sh <checkpoint> maze",
        help="Command shown in the generated markdown report.",
    )
    return parser.parse_args()


def read_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as raw:
        first = raw.readline()
    if first.startswith("version https://git-lfs.github.com/spec/v1"):
        raise ValueError(f"Dataset is a Git LFS pointer: {path}")
    with jsonlines.open(path) as reader:
        for idx, row in enumerate(reader):
            if max_samples is not None and idx >= max_samples:
                break
            rows.append(row)
    return rows


def load_ssvr_model(
    checkpoint: str,
    base_model: str,
    dtype: torch.dtype,
    device: str,
    backbone_type: str,
):
    if backbone_type == "qwen25vl":
        backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=dtype,
            use_safetensors=True,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    elif backbone_type == "lvm":
        backbone = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            use_safetensors=True,
            attn_implementation="eager",
        )
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}")
    adapter_dir = Path(checkpoint) / "backbone"
    if adapter_dir.exists():
        backbone = PeftModel.from_pretrained(backbone, str(adapter_dir), torch_dtype=dtype)
    model = load_local_action_model(backbone, checkpoint, map_location="cpu")
    model.to(device)
    model.eval()
    return model


def build_lvm_inputs(
    sample: Mapping[str, Any],
    start_tokens_index: Mapping[Any, Any],
    image_root: str,
    image_scale: float,
    device: str,
) -> Tuple[Dict[str, torch.Tensor], Image.Image, str]:
    tokens = qwen_visual_tokens_for_sample(sample, start_tokens_index=start_tokens_index)
    image = qwen_visual_image_from_tokens(image_root, tokens, scale=image_scale)
    return {
        "global_input_ids": torch.tensor([tokens], dtype=torch.long, device=device),
    }, image.convert("RGB"), "VQ visual tokens + explicit state token"


def qwen_module(model: Any) -> Any:
    if hasattr(model, "_qwen_module"):
        return model._qwen_module()
    backbone = model.backbone
    return getattr(backbone, "base_model", backbone)


def build_qwen_inputs(
    processor: Any,
    sample: Mapping[str, Any],
    start_tokens_index: Mapping[Any, Any],
    image_root: str,
    image_scale: float,
    device: str,
) -> Tuple[Dict[str, torch.Tensor], Image.Image, str]:
    meta = sample["meta"]
    prompt_text = qwen_visual_prompt_text(meta, normalize_state(sample["input_state"], meta))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = qwen_visual_image_from_tokens(
        image_root,
        qwen_visual_tokens_for_sample(sample, start_tokens_index=start_tokens_index),
        scale=image_scale,
    )
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    return {
        "qwen_input_ids": inputs["input_ids"].to(device),
        "qwen_attention_mask": inputs["attention_mask"].to(device),
        "qwen_pixel_values": inputs["pixel_values"].to(device),
        "qwen_image_grid_thw": inputs["image_grid_thw"].to(device),
    }, image.convert("RGB"), prompt_text


def image_grid_shape(qwen: Any, image_grid_thw: torch.Tensor, num_image_tokens: int) -> Tuple[int, int, str]:
    grid = [int(v) for v in image_grid_thw[0].detach().cpu().tolist()]
    t, h, w = grid
    merge = int(getattr(getattr(qwen.config, "vision_config", None), "spatial_merge_size", 2) or 2)
    candidates = [
        (max(1, h // merge), max(1, w // merge), f"grid_h/w_div_merge{merge}"),
        (h, w, "raw_grid_h/w"),
    ]
    for cand_h, cand_w, mode in candidates:
        if t * cand_h * cand_w == int(num_image_tokens):
            return cand_h, cand_w, mode
    side = int(math.isqrt(int(num_image_tokens)))
    if side * side == int(num_image_tokens):
        return side, side, "square_fallback"
    best_h = side
    while best_h > 1 and int(num_image_tokens) % best_h != 0:
        best_h -= 1
    return best_h, int(num_image_tokens) // best_h, "rectangular_fallback"


def attention_layer_indices(num_layers: int, layer: str) -> List[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if layer == "all":
        return list(range(num_layers))
    if layer == "last":
        return [num_layers - 1]
    index = int(layer)
    if index < 0:
        index += num_layers
    if index < 0 or index >= num_layers:
        raise IndexError(f"attention layer index out of range: {layer}; num_layers={num_layers}")
    return [index]


def head_reduce(values: torch.Tensor, head: str) -> torch.Tensor:
    if values.dim() != 2:
        raise ValueError(f"head_reduce expects [heads, tokens], got {tuple(values.shape)}")
    if head == "mean":
        return values.mean(dim=0)
    index = int(head)
    if index < 0:
        index += values.size(0)
    if index < 0 or index >= values.size(0):
        raise IndexError(f"attention head index out of range: {head}; num_heads={values.size(0)}")
    return values[index]


def attribution_from_attention_tensors(
    attentions: Sequence[torch.Tensor],
    image_positions: torch.Tensor,
    mode: str,
    layer: str,
    head: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if not attentions:
        raise RuntimeError("Qwen did not return attentions; ensure attn_implementation='eager'.")
    selected_layers = attention_layer_indices(len(attentions), layer)
    per_layer_scores: List[torch.Tensor] = []
    used_gradients = False
    fallback_reason = "none"

    for idx in selected_layers:
        attention = attentions[idx]
        state_to_image = attention[0, :, -1, :].index_select(
            dim=-1,
            index=image_positions.to(attention.device),
        ).float()
        if mode == "raw_attention":
            score = head_reduce(state_to_image, head)
        elif mode == "grad_x_attention":
            if attention.grad is None:
                fallback_reason = "missing_attention_gradient"
                score = head_reduce(state_to_image, head)
            else:
                state_grad = attention.grad[0, :, -1, :].index_select(
                    dim=-1,
                    index=image_positions.to(attention.grad.device),
                ).float()
                grad_attention = (state_to_image * state_grad).clamp_min(0.0)
                if float(grad_attention.detach().sum().cpu()) <= 0.0:
                    fallback_reason = "nonpositive_grad_x_attention"
                    grad_attention = (state_to_image * state_grad).abs()
                score = head_reduce(grad_attention, head)
                used_gradients = True
        else:
            raise ValueError(f"Unsupported attribution mode: {mode}")
        per_layer_scores.append(score)

    attribution = torch.stack(per_layer_scores, dim=0).sum(dim=0)
    if float(attribution.detach().sum().cpu()) <= 0.0:
        fallback_reason = "zero_attribution_fallback_to_raw"
        raw_scores: List[torch.Tensor] = []
        for idx in selected_layers:
            raw_scores.append(
                head_reduce(
                    attentions[idx][0, :, -1, :].index_select(
                        dim=-1,
                        index=image_positions.to(attentions[idx].device),
                    ).float(),
                    head,
                )
            )
        attribution = torch.stack(raw_scores, dim=0).sum(dim=0)
    return attribution.detach().float().cpu(), {
        "attribution_mode": mode,
        "attention_layers": selected_layers,
        "attention_head": head,
        "used_attention_gradients": bool(used_gradients),
        "fallback_reason": fallback_reason,
    }


def forward_qwen_attention(
    model: Any,
    qwen_inputs: Mapping[str, torch.Tensor],
    position: torch.Tensor,
    map_size: torch.Tensor,
    state_flags: Optional[torch.Tensor],
    latent_state: Optional[torch.Tensor],
    attribution_mode: str,
    attn_layer: str,
    attn_head: str,
    action_mask: Optional[torch.Tensor] = None,
    attribution_action_id: Optional[int] = None,
) -> Dict[str, Any]:
    if hasattr(model, "zero_grad"):
        model.zero_grad(set_to_none=True)
    grad_enabled = attribution_mode == "grad_x_attention"
    qwen = qwen_module(model)
    input_ids = qwen_inputs["qwen_input_ids"].to(position.device)
    attention_mask = qwen_inputs["qwen_attention_mask"].to(position.device)
    pixel_values = qwen_inputs["qwen_pixel_values"].to(position.device).type(qwen.visual.dtype)
    image_grid_thw = qwen_inputs["qwen_image_grid_thw"].to(position.device)

    with torch.no_grad():
        inputs_embeds = qwen.model.embed_tokens(input_ids)
        image_embeds = qwen.visual(pixel_values, grid_thw=image_grid_thw)
    image_token_id = int(qwen.config.image_token_id)
    image_token_mask_1d = input_ids[0] == image_token_id
    image_mask = image_token_mask_1d.unsqueeze(0).unsqueeze(-1).expand_as(inputs_embeds)
    if int(image_token_mask_1d.sum().item()) != int(image_embeds.shape[0]):
        raise ValueError(
            "Qwen image token/features mismatch: "
            f"tokens={int(image_token_mask_1d.sum().item())}, features={int(image_embeds.shape[0])}"
        )
    inputs_embeds = inputs_embeds.masked_scatter(
        image_mask.to(inputs_embeds.device),
        image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
    )
    inputs_embeds = inputs_embeds.detach()
    inputs_embeds.requires_grad_(grad_enabled)

    if latent_state is None:
        raise ValueError("latent_state is required for implicit-state attribution")
    state_embedding = model.encode_state(
        position, map_size, state_flags, latent_state=latent_state
    ).to(
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    ).detach()
    state_embedding.requires_grad_(grad_enabled)
    inputs_embeds = torch.cat((inputs_embeds, state_embedding.unsqueeze(1)), dim=1)
    state_attention = torch.ones(
        (attention_mask.size(0), 1),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    full_attention_mask = torch.cat((attention_mask, state_attention), dim=1)

    position_ids, _ = qwen.get_rope_index(input_ids, image_grid_thw, None, None, attention_mask)
    valid_lengths = attention_mask.long().sum(dim=1).clamp_min(1)
    batch_indices = torch.arange(position_ids.size(1), device=position_ids.device)
    last_indices = (valid_lengths - 1).to(device=position_ids.device)
    state_position = position_ids[:, batch_indices, last_indices] + 1
    position_ids = torch.cat((position_ids, state_position.unsqueeze(-1)), dim=-1)

    with torch.enable_grad() if grad_enabled else torch.no_grad():
        outputs = qwen.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=full_attention_mask,
            past_key_values=None,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )
        if grad_enabled:
            for attention in outputs.attentions:
                if attention.requires_grad:
                    attention.retain_grad()
        last_hidden = model._select_readout_hidden(
            outputs.last_hidden_state,
            torch.full(
                (outputs.last_hidden_state.size(0),),
                outputs.last_hidden_state.size(1) - 1,
                dtype=torch.long,
                device=outputs.last_hidden_state.device,
            ),
        )
        logits = model._action_logits_from_hidden(last_hidden)
        logits = apply_action_mask_to_logits(logits, action_mask)
        probs = torch.softmax(logits.float(), dim=-1)
        action_id = int(torch.argmax(probs, dim=-1).item())
        target_action_id = action_id if attribution_action_id is None else int(attribution_action_id)
        if target_action_id < 0 or target_action_id >= logits.size(-1):
            raise ValueError(
                f"attribution_action_id out of range: {target_action_id}; "
                f"num_actions={logits.size(-1)}"
            )
        if grad_enabled:
            logits.float()[0, target_action_id].backward()

    image_positions = torch.where(image_token_mask_1d)[0].to(device=input_ids.device)
    image_attention, attribution_info = attribution_from_attention_tensors(
        outputs.attentions,
        image_positions=image_positions,
        mode=attribution_mode,
        layer=attn_layer,
        head=attn_head,
    )
    grid_h, grid_w, grid_mode = image_grid_shape(qwen, image_grid_thw, int(image_attention.numel()))
    heatmap = image_attention.float().cpu().numpy().reshape(grid_h, grid_w)
    heatmap = heatmap - float(heatmap.min())
    max_value = float(heatmap.max())
    if max_value > 0.0:
        heatmap = heatmap / max_value
    if hasattr(model, "zero_grad"):
        model.zero_grad(set_to_none=True)
    return {
        "action_id": action_id,
        "attribution_action_id": target_action_id,
        "logits": logits[0].detach().float().cpu().tolist(),
        "probs": probs[0].detach().float().cpu().tolist(),
        "heatmap": heatmap,
        "attribution_info": attribution_info,
        "image_grid_mode": grid_mode,
        "num_image_tokens": int(image_attention.numel()),
        "image_grid_thw": image_grid_thw[0].detach().cpu().tolist(),
    }


def forward_lvm_attention(
    model: Any,
    lvm_inputs: Mapping[str, torch.Tensor],
    position: torch.Tensor,
    map_size: torch.Tensor,
    state_flags: Optional[torch.Tensor],
    latent_state: Optional[torch.Tensor],
    attribution_mode: str,
    attn_layer: str,
    attn_head: str,
    action_mask: Optional[torch.Tensor] = None,
    attribution_action_id: Optional[int] = None,
) -> Dict[str, Any]:
    if hasattr(model, "zero_grad"):
        model.zero_grad(set_to_none=True)
    grad_enabled = attribution_mode == "grad_x_attention"
    global_input_ids = lvm_inputs["global_input_ids"].to(position.device)
    global_sequence = model.build_global_sequence(global_input_ids)
    inputs_embeds = model.backbone.get_input_embeddings()(global_sequence).detach()
    inputs_embeds.requires_grad_(grad_enabled)
    attention_mask = torch.ones_like(global_sequence, dtype=torch.long)
    if latent_state is None:
        raise ValueError("latent_state is required for implicit-state attribution")
    state_embedding = model.encode_state(
        position, map_size, state_flags, latent_state=latent_state
    ).to(
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    ).detach()
    state_embedding.requires_grad_(grad_enabled)
    inputs_embeds, attention_mask, readout_indices = model._fuse_state_embedding(
        inputs_embeds,
        attention_mask,
        state_embedding,
    )

    with torch.enable_grad() if grad_enabled else torch.no_grad():
        outputs = model.backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        if grad_enabled:
            for attention in outputs.attentions:
                if attention.requires_grad:
                    attention.retain_grad()
        last_hidden = model._select_readout_hidden(outputs.hidden_states[-1], readout_indices)
        logits = model._action_logits_from_hidden(last_hidden)
        logits = apply_action_mask_to_logits(logits, action_mask)
        probs = torch.softmax(logits.float(), dim=-1)
        action_id = int(torch.argmax(probs, dim=-1).item())
        target_action_id = action_id if attribution_action_id is None else int(attribution_action_id)
        if target_action_id < 0 or target_action_id >= logits.size(-1):
            raise ValueError(
                f"attribution_action_id out of range: {target_action_id}; "
                f"num_actions={logits.size(-1)}"
            )
        if grad_enabled:
            logits.float()[0, target_action_id].backward()


    image_positions = torch.arange(
        1,
        1 + global_input_ids.size(1),
        dtype=torch.long,
        device=global_input_ids.device,
    )
    image_attention, attribution_info = attribution_from_attention_tensors(
        outputs.attentions,
        image_positions=image_positions,
        mode=attribution_mode,
        layer=attn_layer,
        head=attn_head,
    )
    num_tokens = int(image_attention.numel())
    side = int(math.isqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"LVM VQ token count must form a square grid, got {num_tokens}")
    heatmap = image_attention.numpy().reshape(side, side)
    heatmap = heatmap - float(heatmap.min())
    max_value = float(heatmap.max())
    if max_value > 0.0:
        heatmap = heatmap / max_value
    if hasattr(model, "zero_grad"):
        model.zero_grad(set_to_none=True)
    return {
        "action_id": action_id,
        "attribution_action_id": target_action_id,
        "logits": logits[0].detach().float().cpu().tolist(),
        "probs": probs[0].detach().float().cpu().tolist(),
        "heatmap": heatmap,
        "attribution_info": attribution_info,
        "image_grid_mode": f"lvm_vq_{side}x{side}",
        "num_image_tokens": num_tokens,
        "image_grid_thw": [1, side, side],
    }


@torch.no_grad()
def forward_implicit_action(
    model: Any,
    model_inputs: Mapping[str, torch.Tensor],
    global_input_ids: torch.Tensor,
    position: torch.Tensor,
    map_size: torch.Tensor,
    state_flags: Optional[torch.Tensor],
    latent_state: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:


    forward_kwargs = {
        key: value
        for key, value in model_inputs.items()
        if key.startswith("qwen_")
    }
    outputs = model(
        global_input_ids=global_input_ids,
        position=position,
        map_size=map_size,
        state_flags=state_flags,
        latent_state=latent_state,
        **forward_kwargs,
    )
    logits = apply_action_mask_to_logits(outputs.logits, action_mask)
    probs = torch.softmax(logits.float(), dim=-1)
    return {
        "action_id": int(torch.argmax(probs, dim=-1).item()),
        "logits": logits[0].detach().float().cpu().tolist(),
        "probs": probs[0].detach().float().cpu().tolist(),
        "heatmap": None,
        "attribution_info": {},
        "image_grid_mode": "not_collected",
        "num_image_tokens": 0,
        "image_grid_thw": [],
    }


def rollout_case(
    model: Any,
    processor: Optional[Any],
    sample: Mapping[str, Any],
    start_tokens_index: Mapping[Any, Any],
    image_root: str,
    image_scale: float,
    device: str,
    max_steps: Optional[int],
    minibehaviour_interaction_action_masking: bool = False,
    collect_attention: bool = False,
    output_case_dir: Optional[Path] = None,
    attribution_mode: str = "grad_x_attention",
    attn_layer: str = "all",
    attn_head: str = "mean",
    minibehaviour_split_attribution: bool = False,
    alpha: float = 0.45,
    dpi: int = 160,
    backbone_type: str = "qwen25vl",
) -> Dict[str, Any]:
    meta = sample["meta"]
    level = int(meta["level"])
    height = int(meta.get("height", level))
    width = int(meta.get("width", level))
    start_state = normalize_state(sample["input_state"], meta)
    expected = task_distance_to_goal(meta, start_state)
    if expected is None:
        raise ValueError(f"Missing distance for start_state={start_state}")
    expected_move = int(expected)
    total_steps = int(max_steps) if max_steps is not None else max(1, expected_move)
    map_size = torch.tensor([[height, width]], dtype=torch.float32, device=device)
    model_map_size = torch.ones_like(map_size) if is_maze_flip_meta(meta) else map_size
    if backbone_type == "qwen25vl":
        if processor is None:
            raise ValueError("Qwen attention visualization requires a processor")
        model_inputs, image, prompt_text = build_qwen_inputs(
            processor, sample, start_tokens_index, image_root, image_scale, device
        )
    else:
        model_inputs, image, prompt_text = build_lvm_inputs(
            sample, start_tokens_index, image_root, image_scale, device
        )

    original_image_path = None
    if collect_attention and output_case_dir is not None:
        original_image_path = output_case_dir / "input_image.png"
        original_image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(original_image_path)

    current_state = start_state
    global_input_ids = torch.tensor(
        [qwen_visual_tokens_for_sample(sample, start_tokens_index=start_tokens_index)],
        dtype=torch.long,
        device=device,
    )
    latent_state = model.initialize_latent_state(global_input_ids)
    coords = [state_to_task_position(meta, current_state)]
    steps: List[Dict[str, Any]] = []
    action_ids: List[int] = []
    termination_reason = "none"
    for step in range(total_steps):
        row, col = state_to_task_position(meta, current_state)
        position = torch.tensor([[row, col]], dtype=torch.float32, device=device)
        model_position = torch.zeros_like(position) if is_maze_flip_meta(meta) else position
        state_flags = None
        if getattr(model, "num_state_flags", 0) > 0:
            state_flags = torch.tensor(
                [[float(state_carrying(meta, current_state))]],
                dtype=torch.float32,
                device=device,
            )
            if is_maze_flip_meta(meta):
                state_flags = torch.zeros_like(state_flags)
        action_mask = None
        if minibehaviour_interaction_action_masking and is_minibehaviour_meta(meta):
            action_mask = torch.tensor(
                [minibehaviour_interaction_action_mask(meta, current_state)],
                dtype=torch.float32,
                device=device,
            )
        forward_attention = (
            forward_qwen_attention if backbone_type == "qwen25vl" else forward_lvm_attention
        )
        if collect_attention:
            output = forward_attention(
                model,
                model_inputs,
                position=model_position,
                map_size=model_map_size,
                state_flags=state_flags,
                latent_state=latent_state,
                attribution_mode=attribution_mode,
                attn_layer=attn_layer,
                attn_head=attn_head,
                action_mask=action_mask,
            )
        else:
            output = forward_implicit_action(
                model,
                model_inputs,
                global_input_ids=global_input_ids,
                position=model_position,
                map_size=model_map_size,
                state_flags=state_flags,
                latent_state=latent_state,
                action_mask=action_mask,
            )
        action_id = int(output["action_id"])
        action_names = action_names_for_meta(meta)
        action_name = str(action_names[action_id])
        world_action_id = maze_flip_view_action_to_world(meta, current_state, action_id)
        world_action_name = str(action_names[world_action_id])
        transition = transition_step(meta, current_state, world_action_id)
        action_ids.append(action_id)
        heatmap_path = None
        split_heatmap_paths: Dict[str, str] = {}
        split_attribution_info: Dict[str, Any] = {}
        if collect_attention and output_case_dir is not None:
            heatmap_path = output_case_dir / f"step_{step:02d}_{action_name.lower()}_attention.png"
            save_attention_overlay(
                image=image,
                heatmap=output["heatmap"],
                path=heatmap_path,
                title="",
                alpha=alpha,
                dpi=dpi,
            )
            if minibehaviour_split_attribution and is_minibehaviour_meta(meta):
                name_to_id = {str(name).upper(): idx for idx, name in enumerate(action_names)}
                move_ids = [
                    idx
                    for idx, name in enumerate(action_names)
                    if str(name).upper() in {"UP", "DOWN", "LEFT", "RIGHT"}
                ]
                split_targets: List[Tuple[str, int]] = []
                if move_ids:
                    nav_id = max(move_ids, key=lambda idx: float(output["logits"][idx]))
                    split_targets.append(("navigation", int(nav_id)))
                for target_name in ("PICK", "DROP"):
                    if target_name in name_to_id:
                        split_targets.append((target_name.lower(), int(name_to_id[target_name])))
                for target_label, target_id in split_targets:
                    split_output = forward_attention(
                        model,
                        model_inputs,
                        position=model_position,
                        map_size=model_map_size,
                        state_flags=state_flags,
                        latent_state=latent_state,
                        attribution_mode=attribution_mode,
                        attn_layer=attn_layer,
                        attn_head=attn_head,
                        action_mask=action_mask,
                        attribution_action_id=target_id,
                    )
                    split_path = (
                        output_case_dir
                        / f"step_{step:02d}_{target_label}_target_attention.png"
                    )
                    save_attention_overlay(
                        image=image,
                        heatmap=split_output["heatmap"],
                        path=split_path,
                        title="",
                        alpha=alpha,
                        dpi=dpi,
                    )
                    split_heatmap_paths[target_label] = str(split_path)
                    split_attribution_info[target_label] = {
                        **split_output["attribution_info"],
                        "target_action": str(action_names[target_id]),
                        "target_action_id": int(target_id),
                    }
        steps.append(
            {
                "step": int(step),
                "state": int(current_state),
                "position": [int(row), int(col)],
                "action_id": int(action_id),
                "action": action_name,
                "world_action_id": int(world_action_id),
                "world_action": world_action_name,
                "valid": bool(transition.valid),
                "reason": transition.reason,
                "next_state": int(transition.next_state),
                "probs": {
                    str(action_names[idx]): float(prob)
                    for idx, prob in enumerate(output["probs"])
                },
                "heatmap_path": str(heatmap_path) if heatmap_path is not None else None,
                "split_heatmap_paths": split_heatmap_paths,
                "attribution_info": output["attribution_info"],
                "split_attribution_info": split_attribution_info,
                "image_grid_mode": output["image_grid_mode"],
                "num_image_tokens": int(output["num_image_tokens"]),
                "image_grid_thw": output["image_grid_thw"],
            }
        )
        latent_state = model.update_latent_state(
            latent_state,
            torch.tensor([action_id], dtype=torch.long, device=latent_state.device),
        )
        current_state = int(transition.next_state)
        coords.append(state_to_task_position(meta, current_state))
        if transition.terminal:
            termination_reason = transition.reason
            break

    target_pos = int(meta.get("target_pos", -1))
    all_valid = all(bool(step["valid"]) for step in steps[:expected_move])
    if is_minibehaviour_meta(meta):
        complete = bool(termination_reason == "target" and all_valid)
        optimal_prefix = 0
        replay_state = start_state
        for action_id in action_ids[:expected_move]:
            before = task_distance_to_goal(meta, replay_state)
            transition = transition_step(meta, replay_state, int(action_id))
            if not transition.valid:
                break
            after = (
                0.0
                if transition.terminal and transition.reason == "target"
                else task_distance_to_goal(meta, transition.next_state)
            )
            if before is None or after is None or after != before - 1:
                break
            optimal_prefix += 1
            replay_state = int(transition.next_state)
        em = 1.0 if complete and optimal_prefix == expected_move else 0.0
        pr = optimal_prefix / expected_move if expected_move else 1.0
    else:
        em, pr = compute_em_pr(
            start_coords=[[int(r), int(c)] for r, c in coords],
            expected_move=expected_move,
            distance_map=meta.get("distance_map", {}),
            target_pos=target_pos,
            width=width,
        )
        complete = bool(termination_reason == "target" and em == 1.0)
    return {
        "level": level,
        "start_state": int(start_state),
        "start_position": [int(coords[0][0]), int(coords[0][1])],
        "target_pos": target_pos,
        "expected_move": expected_move,
        "coords": [[int(r), int(c)] for r, c in coords],
        "steps": steps,
        "em": float(em),
        "pr": float(pr),
        "complete": complete,
        "termination_reason": termination_reason,
        "prompt": prompt_text,
        "original_image_path": str(original_image_path) if original_image_path is not None else None,
    }


def save_attention_overlay(image: Image.Image, heatmap: np.ndarray, path: Path, title: str, alpha: float, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = image.convert("RGBA")
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = heatmap - float(np.nanmin(heatmap))
    max_value = float(np.nanmax(heatmap))
    if max_value > 0.0:
        heatmap = heatmap / max_value
    color = plt.get_cmap("magma")(heatmap)
    color[..., 3] = float(alpha)
    overlay = Image.fromarray((color * 255.0).astype(np.uint8), mode="RGBA")
    overlay = overlay.resize(base.size, Image.Resampling.BILINEAR)
    Image.alpha_composite(base, overlay).convert("RGB").save(path)


def select_cases(
    model: Any,
    processor: Optional[Any],
    samples: Sequence[Mapping[str, Any]],
    sample_indices: Sequence[int],
    start_tokens_index: Mapping[Any, Any],
    levels: Sequence[int],
    image_root: str,
    image_scale: float,
    device: str,
    max_steps: Optional[int],
    cases_per_bucket: int,
    min_correct_move_steps: int = 0,
    minibehaviour_interaction_action_masking: bool = False,
    backbone_type: str = "qwen25vl",
) -> Tuple[Dict[int, Dict[str, List[int]]], List[Dict[str, Any]]]:
    cases_per_bucket = max(1, int(cases_per_bucket))
    min_correct_move_steps = max(0, int(min_correct_move_steps))
    selected: Dict[int, Dict[str, List[int]]] = {int(level): {} for level in levels}
    scan_rows: List[Dict[str, Any]] = []
    for local_idx, sample in enumerate(tqdm(samples, desc="Scanning correct/wrong cases")):
        idx = int(sample_indices[local_idx])
        level = int(sample["meta"]["level"])
        if level not in selected:
            continue
        if (
            len(selected[level].get("correct", [])) >= cases_per_bucket
            and len(selected[level].get("wrong", [])) >= cases_per_bucket
        ):
            if all(
                len(bucket.get("correct", [])) >= cases_per_bucket
                and len(bucket.get("wrong", [])) >= cases_per_bucket
                for bucket in selected.values()
            ):
                break
            continue
        result = rollout_case(
            model,
            processor,
            sample,
            start_tokens_index,
            image_root,
            image_scale,
            device,
            max_steps,
            minibehaviour_interaction_action_masking=minibehaviour_interaction_action_masking,
            collect_attention=False,
            attribution_mode="raw_attention",
            backbone_type=backbone_type,
        )
        action_sequence = [str(step.get("action", "")) for step in result.get("steps", [])]
        movement_steps = sum(action in {"UP", "DOWN", "LEFT", "RIGHT"} for action in action_sequence)
        interaction_steps = sum(action in {"PICK", "DROP"} for action in action_sequence)
        is_correct = float(result["em"]) == 1.0
        if is_correct and movement_steps < min_correct_move_steps:
            bucket_name = "correct_no_movement"
        else:
            bucket_name = "correct" if is_correct else "wrong"
        scan_rows.append(
            {
                "sample_index": idx,
                "level": level,
                "bucket": bucket_name,
                "em": result["em"],
                "pr": result["pr"],
                "termination_reason": result["termination_reason"],
                "expected_move": result["expected_move"],
                "movement_steps": movement_steps,
                "interaction_steps": interaction_steps,
                "action_sequence": " ".join(action_sequence),
            }
        )
        if bucket_name == "correct_no_movement":
            continue
        bucket = selected[level].setdefault(bucket_name, [])
        if len(bucket) < cases_per_bucket:
            bucket.append(int(idx))
    return selected, scan_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_selected_cases(path: Path) -> Dict[int, Dict[str, List[int]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    selected: Dict[int, Dict[str, List[int]]] = {}
    for level, bucket in raw.items():
        selected[int(level)] = {}
        for name, value in dict(bucket).items():
            if isinstance(value, list):
                selected[int(level)][str(name)] = [int(index) for index in value]
            else:
                selected[int(level)][str(name)] = [int(value)]
    return selected


def rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve().parent).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_markdown(
    path: Path,
    args: argparse.Namespace,
    selected: Mapping[int, Mapping[str, Sequence[int]]],
    case_summaries: Sequence[Mapping[str, Any]],
) -> None:
    lines: List[str] = []
    backbone_label = "LVM" if args.backbone_type == "lvm" else "Qwen"
    lines.append(f"# SSVR {backbone_label} Image-Token Attention Visualization")
    lines.append("")
    lines.append("## Experiment settings")
    lines.append("")
    lines.append(f"- Checkpoint: `{args.checkpoint}`")
    lines.append(f"- Backbone type: `{args.backbone_type}`")
    lines.append(f"- Base model: `{args.base_model}`")
    lines.append(f"- Dataset: `{args.test_dataset}`")
    lines.append(f"- Decoded visual image root: `{args.image_root}`")
    lines.append(
        f"- Attribution: `{args.attribution_mode}`; latent-state-token query to image tokens; "
        f"layer=`{args.attn_layer}`, head=`{args.attn_head}`"
    )
    lines.append(f"- MiniBehaviour interaction action masking: `{bool(args.minibehaviour_interaction_action_masking)}`")
    lines.append(f"- MiniBehaviour split attribution: `{bool(args.minibehaviour_split_attribution)}`")
    lines.append(f"- Minimum movement steps for correct-case selection: `{int(args.min_correct_move_steps)}`")
    lines.append("- Heatmap is min-max normalized per step and overlaid on the VQ-VAE decoded initial map image.")
    lines.append(
        "- `grad_x_attention` means `attention * d(predicted_action_logit) / d(attention)` "
        "and is action-specific; `raw_attention` is only a diagnostic baseline."
    )
    if "maze_flip" in str(args.test_dataset).lower():
        lines.append(
            "- MazeFlip rollout treats model outputs as view-frame actions and maps them to "
            "world actions only after prediction; the latent is updated with the view action."
        )
        lines.append(
            "- MazeFlip model-facing position/map/flags are fixed placeholders; no current "
            "flip state or world coordinate is passed to the model."
        )
    lines.append("")
    lines.append("## Reproduction commands")
    lines.append("")
    lines.append("```bash")
    lines.append(str(args.reproduce_command))
    lines.append("```")
    lines.append("")
    lines.append("Alternatively, run directly:")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "/home/usr/miniconda3/envs/visualplanning/bin/python3.12 "
        "tools/visualization/visualize_implicit_attention.py \\"
    )
    lines.append(f"  --checkpoint {args.checkpoint} \\")
    lines.append(f"  --base_model {args.base_model} \\")
    lines.append(f"  --backbone_type {args.backbone_type} \\")
    if args.backbone_type == "qwen25vl":
        lines.append(f"  --processor_path {args.processor_path or args.base_model} \\")
    lines.append(f"  --test_dataset {args.test_dataset} \\")
    lines.append(f"  --image_root {args.image_root} \\")
    lines.append(f"  --output_dir {args.output_dir} \\")
    lines.append(f"  --markdown_path {args.markdown_path} \\")
    lines.append(f"  --attribution_mode {args.attribution_mode} \\")
    lines.append(f"  --attn_layer {args.attn_layer} \\")
    lines.append(f"  --attn_head {args.attn_head} \\")
    if bool(args.minibehaviour_interaction_action_masking):
        lines.append("  --minibehaviour_interaction_action_masking \\")
    if bool(args.minibehaviour_split_attribution):
        lines.append("  --minibehaviour_split_attribution \\")
    lines.append(f"  --min_correct_move_steps {int(args.min_correct_move_steps)} \\")
    lines.append(f"  --cases_per_bucket {args.cases_per_bucket}")
    lines.append("```")
    lines.append("")
    lines.append("## Case selection results")
    lines.append("")
    lines.append("| Level | Correct cases | Wrong cases |")
    lines.append("| ---: | ---: | ---: |")
    for level, bucket in selected.items():
        correct_cases = ", ".join(str(index) for index in bucket.get("correct", [])) or "N/A"
        wrong_cases = ", ".join(str(index) for index in bucket.get("wrong", [])) or "N/A"
        lines.append(
            f"| {level} | {correct_cases} | {wrong_cases} |"
        )
    lines.append("")
    lines.append("## Visualization results")
    lines.append("")
    for case in case_summaries:
        lines.append(
            f"### Level {case['level']} / {case['bucket']} / sample {case['sample_index']}"
        )
        lines.append("")
        lines.append(
            f"- EM: `{case['em']:.4f}`; PR: `{case['pr']:.4f}`; "
            f"expected move: `{case['expected_move']}`; termination: `{case['termination_reason']}`"
        )
        lines.append(f"- Start: `{case['start_position']}`; target state: `{case['target_pos']}`")
        original_image = case.get("original_image_path")
        if original_image:
            lines.append("")
            lines.append(f'<img src="{rel(Path(str(original_image)), path)}" width="240">')
        lines.append("")
        lines.append("| Step | State | Action | Transition | Top probabilities | Attribution | Attention heatmap | Split attribution heatmaps |")
        lines.append("| ---: | --- | --- | --- | --- | --- | --- | --- |")
        for step in case["steps"]:
            heatmap = step.get("heatmap_path")
            heatmap_md = (
                f'<img src="{rel(Path(heatmap), path)}" width="240">'
                if heatmap
                else ""
            )
            probs = sorted(step["probs"].items(), key=lambda item: -float(item[1]))


            shown_probs = probs if len(probs) <= 6 else probs[:4]
            prob_text = ", ".join(f"{name}:{value:.3f}" for name, value in shown_probs)
            transition = f"valid={step['valid']}, reason={step['reason']}, next={step['next_state']}"
            displayed_action = str(step["action"])
            if "world_action" in step:
                displayed_action += f" (world: {step['world_action']})"
            info = step.get("attribution_info", {})
            attribution = (
                f"mode={info.get('attribution_mode', args.attribution_mode)}, "
                f"grad={info.get('used_attention_gradients', False)}, "
                f"fallback={info.get('fallback_reason', 'unknown')}"
            )
            split_parts: List[str] = []
            for target_name, split_path in sorted(step.get("split_heatmap_paths", {}).items()):
                split_info = step.get("split_attribution_info", {}).get(target_name, {})
                target_action = split_info.get("target_action", target_name)
                split_parts.append(
                    f"{target_name}({target_action})<br>"
                    f'<img src="{rel(Path(split_path), path)}" width="180">'
                )
            split_md = "<br>".join(split_parts)
            lines.append(
                f"| {step['step']} | `{step['position']}` | `{displayed_action}` | "
                f"{transition} | {prob_text} | {attribution} | {heatmap_md} | {split_md} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPE_MAP[str(args.torch_dtype)]
    output_dir = Path(args.output_dir)
    markdown_path = Path(args.markdown_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(args.test_dataset, max_samples=args.max_scan_samples)
    num_sample_shards = max(1, int(args.num_sample_shards))
    sample_shard_index = int(args.sample_shard_index)
    if sample_shard_index < 0 or sample_shard_index >= num_sample_shards:
        raise ValueError(
            f"sample_shard_index must be in [0, {num_sample_shards}), got {sample_shard_index}"
        )
    scan_indices = [
        idx for idx in range(len(samples))
        if num_sample_shards == 1 or idx % num_sample_shards == sample_shard_index
    ]
    scan_samples = [samples[idx] for idx in scan_indices]
    levels = (
        [int(part) for part in str(args.levels).split(",") if part.strip()]
        if args.levels
        else sorted({int(sample["meta"]["level"]) for sample in samples})
    )
    processor = None
    if args.backbone_type == "qwen25vl":
        processor = AutoProcessor.from_pretrained(
            args.processor_path or args.base_model,
            trust_remote_code=True,
            use_fast=False,
        )
    start_tokens_index = build_qwen_visual_start_tokens_index(samples)
    model = load_ssvr_model(
        args.checkpoint,
        args.base_model,
        dtype,
        device,
        str(args.backbone_type),
    )

    selected_path = output_dir / "selected_cases.json"
    if selected_path.exists() and not bool(args.rescan_cases):
        selected = load_selected_cases(selected_path)
        scan_rows: List[Dict[str, Any]] = []
        print(f"Reusing selected cases: {selected_path}")
    else:
        selected, scan_rows = select_cases(
            model,
            processor,
            scan_samples,
            scan_indices,
            start_tokens_index,
            levels,
            args.image_root,
            float(args.image_scale),
            device,
            args.max_steps,
            int(args.cases_per_bucket),
            int(args.min_correct_move_steps),
            minibehaviour_interaction_action_masking=bool(args.minibehaviour_interaction_action_masking),
            backbone_type=str(args.backbone_type),
        )
        write_csv(output_dir / "scan_selected_cases.csv", scan_rows)
    if bool(args.select_only):
        (output_dir / "selected_cases.json").write_text(
            json.dumps(selected, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote selected cases: {selected_path}")
        return

    case_summaries: List[Dict[str, Any]] = []
    render_jobs = [
        (int(level), str(bucket), int(sample_index))
        for level in levels
        for bucket in ("correct", "wrong")
        for sample_index in selected.get(level, {}).get(bucket, [])
    ]
    for level, bucket, sample_index in tqdm(render_jobs, desc="Rendering attention cases"):
        case_dir = output_dir / f"level_{level}_{bucket}_sample_{sample_index}"
        result = rollout_case(
            model,
            processor,
            samples[int(sample_index)],
            start_tokens_index,
            args.image_root,
            float(args.image_scale),
            device,
            args.max_steps,
            minibehaviour_interaction_action_masking=bool(args.minibehaviour_interaction_action_masking),
            collect_attention=True,
            output_case_dir=case_dir,
            attribution_mode=str(args.attribution_mode),
            attn_layer=str(args.attn_layer),
            attn_head=str(args.attn_head),
            minibehaviour_split_attribution=bool(args.minibehaviour_split_attribution),
            alpha=float(args.alpha),
            dpi=int(args.dpi),
            backbone_type=str(args.backbone_type),
        )
        result["level"] = int(level)
        result["bucket"] = bucket
        result["sample_index"] = int(sample_index)
        case_summaries.append(result)
        (case_dir / "case_summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    (output_dir / "selected_cases.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "case_summaries.json").write_text(
        json.dumps(case_summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(markdown_path, args, selected, case_summaries)
    print(f"Wrote attention report: {markdown_path}")
    print(f"Wrote attention images: {output_dir}")


if __name__ == "__main__":
    main()
