#!/usr/bin/env python3


from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch
import torch.distributed as dist
from PIL import ImageOps
from tqdm import tqdm
from transformers import AutoProcessor

from ssvr.reasoning import (
    action_names_for_meta,
    qwen_visual_image_from_tokens,
    qwen_visual_prompt_text,
    qwen_visual_tokens_for_sample,
    build_qwen_visual_start_tokens_index,
    normalize_state,
    position_to_state,
    state_carrying,
    state_to_position,
    state_to_task_position,
    task_distance_to_goal,
    transition_step,
    valid_action_mask,
)
from ssvr.evaluation.evaluator import load_local_action_planning_model, read_jsonl_samples
from ssvr.evaluation.metrics import compute_em_pr, summarize_local_action_results


def _dist_info() -> Tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def _transform_components(transform: str) -> Tuple[bool, bool, bool]:
    name = transform.lower()
    return "invert" in name, "vertical" in name or "both" in name, "horizontal" in name or "both" in name


def transform_position(
    position: Sequence[int],
    height: int,
    width: int,
    transform: str,
) -> Tuple[int, int]:
    row, col = int(position[0]), int(position[1])
    _, vertical, horizontal = _transform_components(transform)
    if vertical:
        row = height - 1 - row
    if horizontal:
        col = width - 1 - col
    return row, col


def transform_state(state: int, height: int, width: int, transform: str) -> int:
    row, col = state_to_position(int(state), width)
    row, col = transform_position((row, col), height, width, transform)
    return position_to_state(row, col, width)


def transform_walls(walls: Mapping[str, Any], transform: str) -> Dict[str, bool]:
    _, vertical, horizontal = _transform_components(transform)
    north = bool(walls["north"])
    south = bool(walls["south"])
    west = bool(walls["west"])
    east = bool(walls["east"])
    if vertical:
        north, south = south, north
    if horizontal:
        west, east = east, west
    return {"north": north, "south": south, "west": west, "east": east}


def transform_maze_layout(
    layout: Sequence[Sequence[Mapping[str, Any]]],
    height: int,
    width: int,
    transform: str,
) -> List[List[Dict[str, bool]]]:
    transformed: List[List[Dict[str, bool]]] = [
        [dict(north=False, south=False, west=False, east=False) for _ in range(width)]
        for _ in range(height)
    ]
    for row in range(height):
        for col in range(width):
            new_row, new_col = transform_position((row, col), height, width, transform)
            transformed[new_row][new_col] = transform_walls(layout[row][col], transform)
    return transformed


def transform_distance_map(
    distance_map: Mapping[str, Any],
    height: int,
    width: int,
    transform: str,
) -> Dict[str, Any]:
    return {
        str(transform_state(int(state), height, width, transform)): value
        for state, value in distance_map.items()
    }


def transform_maze_sample(sample: Mapping[str, Any], transform: str) -> Dict[str, Any]:
    transformed = copy.deepcopy(sample)
    meta = transformed["meta"]
    height = int(meta.get("height", meta["level"]))
    width = int(meta.get("width", meta["level"]))
    _, vertical, horizontal = _transform_components(transform)
    if not vertical and not horizontal:
        return transformed

    meta["layout"] = transform_maze_layout(meta["layout"], height, width, transform)
    if "target_pos" in meta:
        meta["target_pos"] = transform_state(int(meta["target_pos"]), height, width, transform)
    if "start_pos" in meta:
        start_pos = meta["start_pos"]
        if isinstance(start_pos, (list, tuple)):
            meta["start_pos"] = list(transform_position(start_pos, height, width, transform))
        else:
            meta["start_pos"] = transform_state(int(start_pos), height, width, transform)
    if "distance_map" in meta:
        meta["distance_map"] = transform_distance_map(meta["distance_map"], height, width, transform)
    transformed["input_state"] = transform_state(
        normalize_state(sample["input_state"], sample["meta"]),
        height,
        width,
        transform,
    )
    if "current_state" in transformed:
        transformed["current_state"] = transformed["input_state"]
    return transformed


def apply_image_transform(image, transform: str):
    invert, vertical, horizontal = _transform_components(transform)
    out = image.convert("RGB")
    if invert:
        out = ImageOps.invert(out)
    if vertical:
        out = ImageOps.flip(out)
    if horizontal:
        out = ImageOps.mirror(out)
    return out


def entropy_from_probs(probs: torch.Tensor) -> float:
    probs = probs.float().clamp_min(1e-12)
    return float(-(probs * probs.log()).sum().item())


@torch.no_grad()
def evaluate_sample(
    model: torch.nn.Module,
    processor: Any,
    original_sample: Mapping[str, Any],
    transformed_sample: Mapping[str, Any],
    start_tokens_index: Mapping[Any, Any],
    image_root: str,
    transform: str,
    image_scale: float,
    device: torch.device,
    use_kv_cache: bool = False,
) -> Dict[str, Any]:
    meta = transformed_sample["meta"]
    level = int(meta["level"])
    height = int(meta.get("height", level))
    width = int(meta.get("width", level))
    start_state = normalize_state(transformed_sample["input_state"], meta)
    expected_distance = task_distance_to_goal(meta, start_state)
    if expected_distance is None:
        raise ValueError(f"Start state {start_state} is absent from transformed distance_map")
    expected_move = int(expected_distance)

    global_input_ids = torch.tensor(
        original_sample["input_tokens"],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    map_size = torch.tensor([[height, width]], dtype=torch.float32, device=device)

    prompt_text = qwen_visual_prompt_text(meta, start_state)
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
        qwen_visual_tokens_for_sample(original_sample, start_tokens_index=start_tokens_index),
        scale=image_scale,
    )
    image = apply_image_transform(image, transform)
    qwen_inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    qwen_visual_tensors = {
        "qwen_input_ids": qwen_inputs["input_ids"].to(device),
        "qwen_attention_mask": qwen_inputs["attention_mask"].to(device),
        "qwen_pixel_values": qwen_inputs["pixel_values"].to(device),
        "qwen_image_grid_thw": qwen_inputs["image_grid_thw"].to(device),
    }

    current_state = start_state
    latent_state = model.initialize_latent_state(global_input_ids)
    qwen_prefix_cache = model.encode_qwen_visual_cache(**qwen_visual_tensors) if use_kv_cache else None
    coords = [state_to_task_position(meta, current_state)]
    action_list: List[str] = []
    logits_history: List[List[float]] = []
    first_step: Dict[str, float] = {}
    complete = False
    action_names = action_names_for_meta(meta)

    for step_idx in range(expected_move):
        row, col = state_to_task_position(meta, current_state)
        position = torch.tensor([[row, col]], dtype=torch.float32, device=device)
        state_step = torch.tensor([step_idx], dtype=torch.float32, device=device)
        state_flags = torch.tensor(
            [[float(state_carrying(meta, current_state))]],
            dtype=torch.float32,
            device=device,
        )
        _, logits = model.predict_action(
            global_input_ids=global_input_ids,
            position=position,
            map_size=map_size,
            use_cache=use_kv_cache,
            state_flags=state_flags,
            state_step=state_step,
            latent_state=latent_state,
            qwen_prefix_cache=qwen_prefix_cache,
            **qwen_visual_tensors,
        )
        action_count = len(action_names)
        probs = torch.softmax(logits.float()[0, :action_count], dim=-1)
        action_id = int(torch.argmax(probs).item())
        latent_state = model.update_latent_state(latent_state, torch.tensor([action_id], device=device))
        result = transition_step(meta, current_state, action_id)
        logits_history.append(logits[0, :action_count].detach().float().cpu().tolist())
        action_list.append(action_names[action_id])

        if step_idx == 0:
            legal_mask = valid_action_mask(meta, current_state)
            legal_ids = [idx for idx, valid in enumerate(legal_mask) if valid > 0.0]
            legal_mass = float(probs[legal_ids].sum().item()) if legal_ids else 0.0
            if legal_ids and legal_mass > 0.0:
                conditional_entropy = entropy_from_probs(probs[legal_ids] / legal_mass)
                normalized_entropy = conditional_entropy / math.log(len(legal_ids)) if len(legal_ids) > 1 else 0.0
            else:
                conditional_entropy = 0.0
                normalized_entropy = 0.0
            current_distance = task_distance_to_goal(meta, current_state)
            next_distance = task_distance_to_goal(meta, result.next_state) if result.valid else None
            closer_or_target = bool(
                result.valid
                and (
                    result.reason == "target"
                    or (
                        current_distance is not None
                        and next_distance is not None
                        and float(next_distance) == float(current_distance) - 1.0
                    )
                )
            )
            first_step = {
                "legal": 1.0 if result.valid else 0.0,
                "hole": 1.0 if result.reason == "hole" else 0.0,
                "entropy": conditional_entropy,
                "normalized_entropy": normalized_entropy,
                "closer_target": 1.0 if closer_or_target else 0.0,
            }

        current_state = result.next_state
        coords.append(state_to_task_position(meta, current_state))
        if result.terminal:
            complete = result.reason == "target" and step_idx == expected_move - 1
            break
        if not result.valid:
            break

    all_valid = len(action_list) >= expected_move
    complete = bool(
        len(coords) > expected_move
        and coords[expected_move] == state_to_position(int(meta["target_pos"]), width)
        and all_valid
    )
    em, pr = compute_em_pr(
        start_coords=[list(coord) for coord in coords],
        expected_move=expected_move,
        distance_map=meta.get("distance_map", {}),
        target_pos=int(meta["target_pos"]),
        width=width,
    )
    return {
        "level": level,
        "complete": complete,
        "em": em,
        "pr": pr,
        "expected_move": expected_move,
        "start_coords": [list(coord) for coord in coords],
        "action_list": action_list,
        "logits": logits_history,
        "first_step": first_step,
        "transform": transform,
    }


def summarize_first_step(results: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    totals = {
        "legal": 0.0,
        "hole": 0.0,
        "entropy": 0.0,
        "normalized_entropy": 0.0,
        "closer_target": 0.0,
    }
    count = 0
    for result in results:
        first = result.get("first_step") or {}
        if not first:
            continue
        count += 1
        for key in totals:
            totals[key] += float(first.get(key, 0.0))
    denom = max(count, 1)
    return {
        "legal_rate": totals["legal"] / denom,
        "hole_rate": totals["hole"] / denom,
        "entropy": totals["entropy"] / denom,
        "normalized_entropy": totals["normalized_entropy"] / denom,
        "closer_target_rate": totals["closer_target"] / denom,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--processor_path", required=True)
    parser.add_argument("--test_dataset", default="dataset/maze/tokenized_dataset/SFT/test_dataset.jsonl")
    parser.add_argument("--image_root", default="output/qwen_visual_initial_maps_vqvae")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--transform", default="none")
    parser.add_argument("--image_scale", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--qwen_processor_use_fast", action="store_true")
    parser.add_argument("--use_kv_cache", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    rank, local_rank, world_size = _dist_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    samples = read_jsonl_samples(args.test_dataset, max_samples=args.max_samples)
    if not samples:
        raise ValueError(f"No samples loaded from {args.test_dataset}")
    first_layout = samples[0]["meta"].get("layout")
    if not (
        isinstance(first_layout, list)
        and first_layout
        and isinstance(first_layout[0], list)
        and first_layout[0]
        and isinstance(first_layout[0][0], Mapping)
    ):
        raise ValueError("This robustness evaluator only supports Maze datasets")

    processor = AutoProcessor.from_pretrained(
        args.processor_path,
        trust_remote_code=True,
        use_fast=bool(args.qwen_processor_use_fast),
    )
    model = load_local_action_planning_model(
        model_path=args.base_model,
        checkpoint_path=args.checkpoint,
        device=str(device),
        torch_dtype=torch.bfloat16,
        backbone_type="qwen25vl",
    )
    start_tokens_index = build_qwen_visual_start_tokens_index(samples)

    local_results: List[Dict[str, Any]] = []
    iterator = range(rank, len(samples), world_size)
    if rank == 0:
        iterator = tqdm(list(iterator), desc=f"Maze transform={args.transform}")
    for sample_idx in iterator:
        original_sample = samples[sample_idx]
        transformed_sample = transform_maze_sample(original_sample, args.transform)
        result = evaluate_sample(
            model=model,
            processor=processor,
            original_sample=original_sample,
            transformed_sample=transformed_sample,
            start_tokens_index=start_tokens_index,
            image_root=args.image_root,
            transform=args.transform,
            image_scale=args.image_scale,
            device=device,
            use_kv_cache=args.use_kv_cache,
        )
        result["sample_index"] = sample_idx
        local_results.append(result)

    gathered: List[List[Dict[str, Any]]] = [None for _ in range(world_size)]
    if world_size > 1:
        dist.all_gather_object(gathered, local_results)
    else:
        gathered = [local_results]

    if rank == 0:
        results = [item for sublist in gathered for item in sublist]
        results.sort(key=lambda row: int(row["sample_index"]))
        summary = summarize_local_action_results(results)
        first_step = summarize_first_step(results)
        row = {
            "checkpoint": args.checkpoint,
            "transform": args.transform,
            "image_scale": args.image_scale,
            "count": summary["overall"]["count"],
            "samples": summary["overall"]["count"],
            "accuracy": summary["overall"]["accuracy"],
            "em": summary["overall"]["em"],
            "pr": summary["overall"]["pr"],
            **first_step,
            "per_level": summary["per_level"],
        }
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps([row], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "case_results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(row, ensure_ascii=False, indent=2))

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
