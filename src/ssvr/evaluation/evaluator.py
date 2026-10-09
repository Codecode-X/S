import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import jsonlines
import torch
from peft import PeftModel
from tqdm import tqdm

from ssvr.backbone import load_backbone_model
from ssvr.reasoning import (
    ACTION_NAMES,
    action_names_for_meta,
    is_minibehaviour_meta,
    load_local_action_model,
    normalize_state,
    qwen_visual_prompt_text,
    qwen_visual_image_from_tokens,
    qwen_visual_tokens_for_sample,
    build_qwen_visual_start_tokens_index,
    state_carrying,
    state_to_task_position,
    state_to_position,
    task_distance_to_goal,
    transition_step,
    maze_flip_view_action_to_world,
    is_maze_flip_meta,
)


log = logging.getLogger(__name__)


def detect_local_action_task(test_dataset_path: str) -> str:
    try:
        with jsonlines.open(test_dataset_path) as reader:
            sample = next(iter(reader))
        meta = sample.get("meta", {})
        if is_minibehaviour_meta(meta):
            return "minibehaviour"
        layout = meta.get("layout")
        if (
            isinstance(layout, list)
            and layout
            and isinstance(layout[0], list)
            and layout[0]
            and isinstance(layout[0][0], Mapping)
        ):
            return "maze"
        if isinstance(layout, list):
            return "frozenlake"
    except (OSError, StopIteration, ValueError):
        pass
    lower = test_dataset_path.lower()
    if "minibehaviour" in lower:
        return "minibehaviour"
    if "maze" in lower:
        return "maze"
    if "frozenlake" in lower or "frozen_lake" in lower:
        return "frozenlake"
    return "unknown"


def load_local_action_planning_model(
    model_path: str,
    checkpoint_path: str,
    device: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    backbone_type: str = "auto",
):
    checkpoint = Path(checkpoint_path)
    adapter_dir = checkpoint / "backbone"
    backbone_path = (
        str(checkpoint)
        if (checkpoint / "config.json").exists() and not adapter_dir.exists()
        else model_path
    )
    backbone = load_backbone_model(
        backbone_path,
        torch_dtype=torch_dtype,
        backbone_type=backbone_type,
    )
    if adapter_dir.exists():
        backbone = PeftModel.from_pretrained(
            backbone,
            str(adapter_dir),
            torch_dtype=torch_dtype,
        )
    model = load_local_action_model(
        backbone=backbone,
        checkpoint_directory=checkpoint_path,
        map_location="cpu",
    )
    model.to(device)
    model.eval()
    return model


def read_jsonl_samples(filepath: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with jsonlines.open(filepath) as reader:
        for idx, obj in enumerate(reader):
            if max_samples is not None and idx >= max_samples:
                break
            samples.append(obj)
    return samples


def _action_tuple(action_id: int, valid: bool, action_names=ACTION_NAMES) -> List[Any]:
    if not valid:
        return [-1, "invalid"]
    return [int(action_id), str(action_names[int(action_id)]).lower()]


def run_local_action_inference_on_sample(
    model,
    sample: Mapping[str, Any],
    device: str,
    task: str,
    max_steps: Optional[int] = None,
    double: bool = False,
    use_cache: bool = True,
    qwen_visual_input: bool = False,
    qwen_processor: Optional[Any] = None,
    qwen_visual_image_size: int = 256,
    qwen_visual_image_root: Optional[str] = None,
    qwen_visual_start_tokens_index: Optional[Mapping[Any, Any]] = None,
    qwen_prompt_text_override: Optional[str] = None,
    qwen_visual_image_scale: float = 1.0,
) -> Dict[str, Any]:
    meta = sample["meta"]
    level = int(meta["level"])
    start_state = normalize_state(sample["input_state"], meta)
    mini = is_minibehaviour_meta(meta)
    target_pos = int(meta.get("target_pos", -1))
    expected_distance = task_distance_to_goal(meta, start_state)
    if expected_distance is None:
        raise ValueError(f"Start state {start_state} is absent from task distance maps")
    expected_move = int(expected_distance)
    num_steps = max_steps if max_steps is not None else expected_move
    if double:
        num_steps *= 2

    global_input_ids = torch.tensor(
        sample["input_tokens"],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    height = int(meta.get("height", level))
    width = int(meta.get("width", level))
    map_size = torch.tensor([[height, width]], dtype=torch.float32, device=device)
    model_map_size = torch.ones_like(map_size) if is_maze_flip_meta(meta) else map_size
    if qwen_visual_input:
        if qwen_processor is None:
            raise ValueError("qwen_processor is required when qwen_visual_input=True")

    current_state = start_state
    latent_state = model.initialize_latent_state(global_input_ids)
    coords = [state_to_task_position(meta, current_state)]
    carrying_history = [state_carrying(meta, current_state)]
    action_list: List[List[Any]] = []
    logits_history: List[List[float]] = []
    termination_reason = "none"

    past_key_values = None
    global_attention_mask = None
    qwen_prefix_cache = None
    if use_cache and not qwen_visual_input:
        past_key_values, global_attention_mask = model.encode_global_cache(global_input_ids)

    for step_idx in range(num_steps):
        row, col = state_to_task_position(meta, current_state)
        position = torch.tensor([[row, col]], dtype=torch.float32, device=device)
        state_step = torch.tensor([step_idx], dtype=torch.float32, device=device)
        state_flags = torch.tensor(
            [[float(state_carrying(meta, current_state))]],
            dtype=torch.float32,
            device=device,
        )
        model_position = torch.zeros_like(position) if is_maze_flip_meta(meta) else position
        model_state_step = torch.zeros_like(state_step) if is_maze_flip_meta(meta) else state_step
        model_state_flags = torch.zeros_like(state_flags) if is_maze_flip_meta(meta) else state_flags
        qwen_visual_tensors: Dict[str, torch.Tensor] = {}
        if qwen_visual_input and (not use_cache or qwen_prefix_cache is None):
            prompt_text = qwen_prompt_text_override or qwen_visual_prompt_text(meta, current_state)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            text = qwen_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if not qwen_visual_image_root:
                raise ValueError(
                    "Qwen native visual inference requires qwen_visual_image_root with "
                    "VQ-decoded initial-map images."
                )
            image = qwen_visual_image_from_tokens(
                qwen_visual_image_root,
                qwen_visual_tokens_for_sample(
                    sample,
                    start_tokens_index=qwen_visual_start_tokens_index,
                ),
                scale=qwen_visual_image_scale,
            )
            qwen_inputs = qwen_processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            qwen_visual_tensors = {
                "qwen_input_ids": qwen_inputs["input_ids"].to(device),
                "qwen_attention_mask": qwen_inputs["attention_mask"].to(device),
                "qwen_pixel_values": qwen_inputs["pixel_values"].to(device),
                "qwen_image_grid_thw": qwen_inputs["image_grid_thw"].to(device),
            }
        with torch.no_grad():
            if use_cache and qwen_visual_input and qwen_prefix_cache is None:
                qwen_prefix_cache = model.encode_qwen_visual_cache(**qwen_visual_tensors)
            if use_cache and not qwen_visual_input:
                outputs = model.forward_with_cache(
                    position=model_position,
                    map_size=model_map_size,
                    past_key_values=past_key_values,
                    global_attention_mask=global_attention_mask,
                    state_flags=model_state_flags,
                    state_step=model_state_step,
                    latent_state=latent_state,
                )
                logits_tensor = outputs.logits
                action_id = int(torch.argmax(logits_tensor, dim=-1).item())
                logits = logits_tensor[0].detach().float().cpu().tolist()
            else:
                action_tensor, logits_tensor = model.predict_action(
                    global_input_ids=global_input_ids,
                    position=model_position,
                    map_size=model_map_size,
                    use_cache=use_cache,
                    state_flags=model_state_flags,
                    state_step=model_state_step,
                    latent_state=latent_state,
                    qwen_prefix_cache=qwen_prefix_cache,
                    **qwen_visual_tensors,
                )
                action_id = int(action_tensor.item())
                logits = logits_tensor[0].detach().float().cpu().tolist()

        latent_state = model.update_latent_state(
            latent_state,
            torch.tensor([action_id], dtype=torch.long, device=device),
        )

        world_action_id = maze_flip_view_action_to_world(meta, current_state, action_id)
        transition = transition_step(meta, current_state, world_action_id)
        action_names = action_names_for_meta(meta)
        action_list.append(_action_tuple(action_id, transition.valid, action_names))
        logits_history.append(logits)
        current_state = transition.next_state
        coords.append(state_to_task_position(meta, current_state))
        carrying_history.append(state_carrying(meta, current_state))
        if transition.terminal:
            termination_reason = transition.reason
            break

    all_valid = all(action[1] != "invalid" for action in action_list[:expected_move])
    if mini:
        complete = termination_reason == "target" and all_valid
        optimal_prefix = 0
        replay_state = start_state
        for action in action_list[:expected_move]:
            if action[1] == "invalid":
                break
            before = task_distance_to_goal(meta, replay_state)
            transition = transition_step(meta, replay_state, int(action[0]))
            after = (
                0.0
                if transition.terminal and transition.reason == "target"
                else task_distance_to_goal(meta, transition.next_state)
            )
            if before is None or after is None or after != before - 1:
                break
            optimal_prefix += 1
            replay_state = transition.next_state
        em = 1.0 if complete and optimal_prefix == expected_move else 0.0
        pr = optimal_prefix / expected_move if expected_move else 1.0
    else:
        complete = (
            len(coords) > expected_move
            and coords[expected_move] == state_to_position(target_pos, width)
            and all_valid
        )
        em = None
        pr = None

    return {
        "start_coords": [list(coord) for coord in coords],
        "action_list": action_list,
        "complete": bool(complete),
        "level": level,
        "start_pos": start_state,
        "target_pos": target_pos,
        "layout": meta.get("layout"),
        "distance_map": meta.get("distance_map", {}),
        "expected_move": expected_move,
        "action_names": list(action_names_for_meta(meta)),
        "logits": logits_history,
        "carrying_history": carrying_history,
        "em": em,
        "pr": pr,
        "task": task,
        "termination_reason": termination_reason,
        "transition_mode": "deterministic_no_slip",
    }


def evaluate_local_action_model(
    model_path: str,
    checkpoint_path: str,
    test_dataset_path: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
    double: bool = False,
    use_cache: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
    backbone_type: str = "auto",
    qwen_visual_input: bool = False,
    qwen_processor_path: Optional[str] = None,
    qwen_visual_image_size: int = 256,
    qwen_processor_use_fast: bool = False,
    qwen_visual_image_root: Optional[str] = None,
    qwen_prompt_text_override: Optional[str] = None,
    qwen_visual_image_scale: float = 1.0,
) -> float:
    task = detect_local_action_task(test_dataset_path)
    if task not in {"frozenlake", "maze", "minibehaviour"}:
        raise NotImplementedError(f"Unsupported local-action task: {task}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_local_action_planning_model(
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        device=device,
        torch_dtype=torch_dtype,
        backbone_type=backbone_type,
    )
    samples = read_jsonl_samples(test_dataset_path, max_samples=max_samples)
    os.makedirs(output_dir, exist_ok=True)
    qwen_processor = None
    if qwen_visual_input:
        from transformers import AutoProcessor

        qwen_processor = AutoProcessor.from_pretrained(
            qwen_processor_path or checkpoint_path,
            trust_remote_code=True,
            use_fast=bool(qwen_processor_use_fast),
        )
    qwen_visual_start_tokens_index = (
        build_qwen_visual_start_tokens_index(samples)
        if qwen_visual_input and qwen_visual_image_root
        else None
    )

    correct_count = 0
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating local-action")):
        result_folder = os.path.join(output_dir, str(idx))
        os.makedirs(result_folder, exist_ok=True)
        json_path = os.path.join(result_folder, "parsed_actions.json")

        result = run_local_action_inference_on_sample(
            model=model,
            sample=sample,
            device=device,
            task=task,
            double=double,
            use_cache=use_cache,
            qwen_visual_input=qwen_visual_input,
            qwen_processor=qwen_processor,
            qwen_visual_image_size=qwen_visual_image_size,
            qwen_visual_image_root=qwen_visual_image_root,
            qwen_visual_start_tokens_index=qwen_visual_start_tokens_index,
            qwen_prompt_text_override=qwen_prompt_text_override,
            qwen_visual_image_scale=qwen_visual_image_scale,
        )
        if result["complete"]:
            correct_count += 1

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

    accuracy = correct_count / len(samples) if samples else 0.0
    log.info("Local-action evaluation accuracy: %.2f%% (%s/%s)", accuracy * 100, correct_count, len(samples))
    return accuracy
