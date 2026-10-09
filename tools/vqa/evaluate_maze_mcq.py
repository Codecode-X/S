#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ssvr.reasoning import (
    ACTION_NAMES,
    build_qwen_visual_start_tokens_index,
    normalize_state,
    qwen_visual_image_from_tokens,
    qwen_visual_tokens_for_sample,
    task_distance_to_goal,
    transition_step,
)


LETTER_TO_ACTION = {"A": 0, "B": 1, "C": 2, "D": 3}
ACTION_TO_LETTER = {value: key for key, value in LETTER_TO_ACTION.items()}
FIRST_LETTER_PATTERN = re.compile(r"^\s*([ABCD])(?:\s|[.)、:：-]|$)", re.IGNORECASE)
ANY_LETTER_PATTERN = re.compile(r"(?:^|[\s:：])([ABCD])(?:\s|[.)、:：-]|$)", re.IGNORECASE)


def read_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= int(max_samples):
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_first_n_per_level(
    samples: Sequence[Mapping[str, Any]],
    samples_per_level: Optional[int],
) -> List[Mapping[str, Any]]:
    if samples_per_level is None:
        return list(samples)
    limit = int(samples_per_level)
    if limit <= 0:
        raise ValueError(f"samples_per_level must be positive, got {limit}")
    counts: Dict[int, int] = defaultdict(int)
    selected: List[Mapping[str, Any]] = []
    for sample in samples:
        level = int(sample["meta"]["level"])
        if counts[level] >= limit:
            continue
        selected.append(sample)
        counts[level] += 1
    return selected


def one_step_prompt() -> str:
    return (
        "Choose the best next move in the maze. Answer with only one letter.\n\n"
        "A. UP\n"
        "B. DOWN\n"
        "C. LEFT\n"
        "D. RIGHT"
    )


def parse_choice(text: str) -> Optional[str]:
    match = FIRST_LETTER_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    match = ANY_LETTER_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    return None


def optimal_action_ids(sample: Mapping[str, Any]) -> List[int]:
    meta = sample["meta"]
    state = normalize_state(sample["input_state"], meta)
    current_distance = task_distance_to_goal(meta, state)
    if current_distance is None:
        return []
    optimal: List[int] = []
    for action_id in range(len(ACTION_NAMES)):
        result = transition_step(meta, state, action_id)
        if not result.valid:
            continue
        next_distance = task_distance_to_goal(meta, result.next_state)
        if next_distance is not None and float(next_distance) == float(current_distance) - 1.0:
            optimal.append(action_id)
    return optimal


def optimal_choice_letter(sample: Mapping[str, Any]) -> Optional[str]:
    optimal_ids = optimal_action_ids(sample)
    if not optimal_ids:
        return None
    return ACTION_TO_LETTER[int(optimal_ids[0])]


def select_few_shot_examples(
    samples: Sequence[Mapping[str, Any]],
    per_action: int,
) -> List[Mapping[str, Any]]:
    if int(per_action) <= 0:
        return []
    grouped: Dict[str, List[Mapping[str, Any]]] = {letter: [] for letter in LETTER_TO_ACTION}
    for sample in samples:
        meta = sample["meta"]
        if int(normalize_state(sample["input_state"], meta)) != int(meta["start_pos"]):
            continue
        letter = optimal_choice_letter(sample)
        if letter is None:
            continue
        if len(grouped[letter]) < int(per_action):
            grouped[letter].append(sample)
        if all(len(rows) >= int(per_action) for rows in grouped.values()):
            break
    missing = {letter: int(per_action) - len(rows) for letter, rows in grouped.items() if len(rows) < int(per_action)}
    if missing:
        raise ValueError(f"Could not find enough few-shot examples per action: {missing}")
    selected: List[Mapping[str, Any]] = []
    for letter in ("A", "B", "C", "D"):
        selected.extend(grouped[letter])
    return selected


def evaluate_choice(sample: Mapping[str, Any], choice: Optional[str]) -> Dict[str, Any]:
    meta = sample["meta"]
    state = normalize_state(sample["input_state"], meta)
    optimal_ids = optimal_action_ids(sample)
    if choice is None:
        return {
            "parse": 0.0,
            "legal": 0.0,
            "closer": 0.0,
            "optimal": 0.0,
            "choice": None,
            "action": None,
            "optimal_actions": [ACTION_NAMES[action_id] for action_id in optimal_ids],
        }
    action_id = LETTER_TO_ACTION[choice]
    result = transition_step(meta, state, action_id)
    current_distance = task_distance_to_goal(meta, state)
    next_distance = task_distance_to_goal(meta, result.next_state) if result.valid else None
    closer = bool(
        result.valid
        and current_distance is not None
        and next_distance is not None
        and float(next_distance) == float(current_distance) - 1.0
    )
    return {
        "parse": 1.0,
        "legal": 1.0 if result.valid else 0.0,
        "closer": 1.0 if closer else 0.0,
        "optimal": 1.0 if action_id in optimal_ids else 0.0,
        "choice": choice,
        "action": ACTION_NAMES[action_id],
        "optimal_actions": [ACTION_NAMES[action_id] for action_id in optimal_ids],
    }


def build_few_shot_messages_and_images(
    sample: Mapping[str, Any],
    few_shot_examples: Sequence[Mapping[str, Any]],
    image_root: str,
    image_scale: float,
    start_tokens_index: Mapping[Any, Sequence[int]],
    few_shot_start_tokens_index: Mapping[Any, Sequence[int]],
) -> tuple[List[Dict[str, Any]], List[Any]]:
    content: List[Dict[str, Any]] = []
    images: List[Any] = []
    for idx, example in enumerate(few_shot_examples):
        letter = optimal_choice_letter(example)
        if letter is None:
            continue
        content.extend(
            [
                {"type": "text", "text": f"Example {idx + 1}. {one_step_prompt()}\n"},
                {"type": "image"},
                {"type": "text", "text": f"\nAnswer: {letter}\n\n"},
            ]
        )
        images.append(
            qwen_visual_image_from_tokens(
                image_root,
                qwen_visual_tokens_for_sample(
                    example,
                    start_tokens_index=few_shot_start_tokens_index,
                ),
                scale=image_scale,
            )
        )
    content.extend(
        [
            {"type": "text", "text": "Now solve this new maze.\n"},
            {"type": "image"},
            {"type": "text", "text": "\n" + one_step_prompt()},
        ]
    )
    images.append(
        qwen_visual_image_from_tokens(
            image_root,
            qwen_visual_tokens_for_sample(sample, start_tokens_index=start_tokens_index),
            scale=image_scale,
        )
    )
    return [{"role": "user", "content": content}], images


def load_model(base_model: str, device: torch.device, torch_dtype: torch.dtype):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model


def attach_adapter(model, checkpoint: str):
    adapter_dir = Path(checkpoint) / "backbone"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing trained backbone adapter: {adapter_dir}")
    return PeftModel.from_pretrained(model, str(adapter_dir)).eval()


@torch.no_grad()
def evaluate_model(
    model,
    processor,
    samples: Sequence[Mapping[str, Any]],
    image_root: str,
    image_scale: float,
    device: torch.device,
    max_new_tokens: int,
    few_shot_examples: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    start_tokens_index = build_qwen_visual_start_tokens_index(samples)
    few_shot_start_tokens_index = (
        build_qwen_visual_start_tokens_index(few_shot_examples) if few_shot_examples else {}
    )
    prompt = one_step_prompt()
    rows: List[Dict[str, Any]] = []
    for local_idx, sample in enumerate(samples):
        meta = sample["meta"]
        if few_shot_examples:
            messages, images = build_few_shot_messages_and_images(
                sample=sample,
                few_shot_examples=few_shot_examples,
                image_root=image_root,
                image_scale=image_scale,
                start_tokens_index=start_tokens_index,
                few_shot_start_tokens_index=few_shot_start_tokens_index,
            )
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            images = [
                qwen_visual_image_from_tokens(
                    image_root,
                    qwen_visual_tokens_for_sample(sample, start_tokens_index=start_tokens_index),
                    scale=image_scale,
                )
            ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(device)
        generated = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        completion = generated[:, inputs["input_ids"].shape[1] :]
        decoded = processor.batch_decode(
            completion,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        choice = parse_choice(decoded)
        metrics = evaluate_choice(sample, choice)
        state = normalize_state(sample["input_state"], meta)
        rows.append(
            {
                "local_index": local_idx,
                "level": int(meta["level"]),
                "start_state": state,
                "target_pos": meta.get("target_pos"),
                "decoded": decoded,
                **metrics,
            }
        )
    return rows


def summarize(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    grouped["overall"] = list(rows)
    for row in rows:
        grouped[f"level_{int(row['level'])}"].append(row)
    summary: List[Dict[str, Any]] = []
    for key in ["overall", "level_3", "level_4", "level_5", "level_6"]:
        group_rows = grouped.get(key, [])
        count = len(group_rows)
        if count == 0:
            continue
        summary.append(
            {
                "group": key,
                "count": count,
                "parse_rate": sum(float(row["parse"]) for row in group_rows) / count,
                "legal_acc": sum(float(row["legal"]) for row in group_rows) / count,
                "closer_acc": sum(float(row["closer"]) for row in group_rows) / count,
                "optimal_acc": sum(float(row["optimal"]) for row in group_rows) / count,
            }
        )
    return summary


def write_csv(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_summary(
    base_summary: Sequence[Mapping[str, Any]],
    trained_summary: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> str:
    base_by_group = {row["group"]: row for row in base_summary}
    trained_by_group = {row["group"]: row for row in trained_summary}
    lines = [
        "# Maze Qwen Native One-Step MCQ Transfer Evaluation",
        "",
        f"- base model: `{args.base_model}`",
        f"- trained local-action checkpoint: `{args.trained_checkpoint}`",
        f"- test dataset: `{args.test_dataset}`",
        f"- max samples: `{args.max_samples if args.max_samples is not None else 'all'}`",
        f"- samples per level: `{args.samples_per_level if args.samples_per_level is not None else 'all'}`",
        f"- few-shot dataset: `{args.few_shot_dataset if args.few_shot_dataset else 'none'}`",
        f"- few-shot examples per action: `{args.few_shot_per_action}`",
        f"- prompt: `{one_step_prompt()}`",
        "- mode: native Qwen `generate()`, one-step A/B/C/D answer, no action head, no state embedding",
        "",
        "| Group | Count | Base Parse | Trained Parse | Base Legal | Trained Legal | Base Closer | Trained Closer | Base Optimal | Trained Optimal | Delta Optimal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in ["overall", "level_3", "level_4", "level_5", "level_6"]:
        if group not in base_by_group or group not in trained_by_group:
            continue
        base = base_by_group[group]
        trained = trained_by_group[group]
        lines.append(
            f"| {group} | {int(trained['count'])} | "
            f"{float(base['parse_rate']):.4f} | {float(trained['parse_rate']):.4f} | "
            f"{float(base['legal_acc']):.4f} | {float(trained['legal_acc']):.4f} | "
            f"{float(base['closer_acc']):.4f} | {float(trained['closer_acc']):.4f} | "
            f"{float(base['optimal_acc']):.4f} | {float(trained['optimal_acc']):.4f} | "
            f"{float(trained['optimal_acc']) - float(base['optimal_acc']):+.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--trained_checkpoint", required=True)
    parser.add_argument("--processor_path", default=None)
    parser.add_argument("--test_dataset", default="dataset/maze/tokenized_dataset/SFT/test_dataset.jsonl")
    parser.add_argument("--image_root", default="output/qwen_visual_initial_maps_vqvae")
    parser.add_argument("--image_scale", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--samples_per_level", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--few_shot_dataset", default=None)
    parser.add_argument("--few_shot_per_action", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--torch_dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--processor_use_fast", action="store_true")
    args = parser.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]

    all_samples = read_jsonl(
        args.test_dataset,
        max_samples=None if args.samples_per_level is not None else args.max_samples,
    )
    all_samples = select_first_n_per_level(all_samples, args.samples_per_level)
    if args.samples_per_level is not None and args.max_samples is not None:
        all_samples = all_samples[: int(args.max_samples)]
    few_shot_examples: List[Mapping[str, Any]] = []
    if args.few_shot_dataset and int(args.few_shot_per_action) > 0:
        few_shot_source = read_jsonl(args.few_shot_dataset)
        few_shot_examples = select_few_shot_examples(
            few_shot_source,
            per_action=int(args.few_shot_per_action),
        )
    rank_samples = [sample for idx, sample in enumerate(all_samples) if idx % world_size == rank]
    processor = AutoProcessor.from_pretrained(
        args.processor_path or args.base_model,
        trust_remote_code=True,
        use_fast=bool(args.processor_use_fast),
    )

    model = load_model(args.base_model, device=device, torch_dtype=dtype)
    base_rows = evaluate_model(
        model=model,
        processor=processor,
        samples=rank_samples,
        image_root=args.image_root,
        image_scale=args.image_scale,
        device=device,
        max_new_tokens=args.max_new_tokens,
        few_shot_examples=few_shot_examples,
    )
    model = attach_adapter(model, args.trained_checkpoint)
    trained_rows = evaluate_model(
        model=model,
        processor=processor,
        samples=rank_samples,
        image_root=args.image_root,
        image_scale=args.image_scale,
        device=device,
        max_new_tokens=args.max_new_tokens,
        few_shot_examples=few_shot_examples,
    )

    gathered_base: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world_size)]
    gathered_trained: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world_size)]
    if distributed:
        dist.all_gather_object(gathered_base, base_rows)
        dist.all_gather_object(gathered_trained, trained_rows)
    else:
        gathered_base = [base_rows]
        gathered_trained = [trained_rows]

    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        merged_base = [row for chunk in gathered_base if chunk for row in chunk]
        merged_trained = [row for chunk in gathered_trained if chunk for row in chunk]
        base_summary = summarize(merged_base)
        trained_summary = summarize(merged_trained)
        write_csv(base_summary, output_dir / "base_summary.csv")
        write_csv(trained_summary, output_dir / "trained_summary.csv")
        write_csv(merged_base, output_dir / "base_cases.csv")
        write_csv(merged_trained, output_dir / "trained_cases.csv")
        if few_shot_examples:
            few_shot_rows = [
                {
                    "index": idx,
                    "level": int(example["meta"]["level"]),
                    "input_state": normalize_state(example["input_state"], example["meta"]),
                    "answer": optimal_choice_letter(example),
                    "action": ACTION_NAMES[LETTER_TO_ACTION[optimal_choice_letter(example)]],
                }
                for idx, example in enumerate(few_shot_examples)
                if optimal_choice_letter(example) is not None
            ]
            write_csv(few_shot_rows, output_dir / "few_shot_examples.csv")
        (output_dir / "base_cases.json").write_text(
            json.dumps(merged_base, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "trained_cases.json").write_text(
            json.dumps(merged_trained, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        md = markdown_summary(base_summary, trained_summary, args)
        (output_dir / "summary.md").write_text(md, encoding="utf-8")
        print(md)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
