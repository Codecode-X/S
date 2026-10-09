from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
import torch.distributed as dist

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ssvr.evaluation.evaluator import load_local_action_planning_model, read_jsonl_samples
from ssvr.training.sft import (
    compute_stage1_closed_loop_metrics,
    compute_stage1_local_action_metrics,
)


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if not match:
        raise ValueError(f"Not a raw checkpoint directory: {path}")
    return int(match.group(1))


def discover_checkpoints(root: Path) -> List[Path]:
    checkpoints = []
    for path in root.iterdir():
        if not path.is_dir() or path.name.endswith("_merged"):
            continue
        if re.fullmatch(r"checkpoint-\d+", path.name):
            checkpoints.append(path)
    return sorted(checkpoints, key=checkpoint_step)


def parse_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def evaluate_one(
    checkpoint: Path,
    base_model: str,
    backbone_type: str,
    samples: List[Mapping[str, Any]],
    device: str,
    torch_dtype: torch.dtype,
    qwen_processor: Optional[Any] = None,
    qwen_visual_image_size: int = 256,
    qwen_visual_image_root: Optional[str] = None,
    qwen_prompt_text_override: Optional[str] = None,
    qwen_visual_image_scale: float = 1.0,
    use_kv_cache: bool = False,
) -> Dict[str, Any]:
    model = load_local_action_planning_model(
        model_path=base_model,
        checkpoint_path=str(checkpoint),
        device=device,
        torch_dtype=torch_dtype,
        backbone_type=backbone_type,
    )
    try:
        local_metrics = compute_stage1_local_action_metrics(
            model=model,
            samples=samples,
            device=torch.device(device),
            use_cache=use_kv_cache,
            qwen_processor=qwen_processor,
            qwen_visual_image_size=qwen_visual_image_size,
            qwen_visual_image_root=qwen_visual_image_root,
            qwen_prompt_text_override=qwen_prompt_text_override,
            qwen_visual_image_scale=qwen_visual_image_scale,
        )
        closed_loop = compute_stage1_closed_loop_metrics(
            model=model,
            samples=samples,
            device=torch.device(device),
            use_cache=use_kv_cache,
            qwen_processor=qwen_processor,
            qwen_visual_image_size=qwen_visual_image_size,
            qwen_visual_image_root=qwen_visual_image_root,
            qwen_prompt_text_override=qwen_prompt_text_override,
            qwen_visual_image_scale=qwen_visual_image_scale,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    step = checkpoint_step(checkpoint)
    return {
        "checkpoint": checkpoint.name,
        "step": step,
        "path": str(checkpoint),
        "samples": int(closed_loop["closed_loop_samples"]),
        "em": float(closed_loop["closed_loop_em"]),
        "pr": float(closed_loop["closed_loop_pr"]),
        "accuracy": float(closed_loop["closed_loop_accuracy"]),
        "legal_rate": float(local_metrics["legal_rate"]),
        "hole_rate": float(local_metrics["hole_rate"]),
        "entropy": float(local_metrics["entropy"]),
        "normalized_entropy": float(local_metrics["normalized_entropy"]),
        "closer_target_rate": float(local_metrics["closer_target_rate"]),
    }


def aggregate_rank_rows(rows: List[Dict[str, Any]], rank: int, world_size: int) -> Optional[List[Dict[str, Any]]]:
    if world_size == 1:
        return rows

    gathered: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, rows)
    if rank != 0:
        return None

    by_step: Dict[int, List[Dict[str, Any]]] = {}
    for rank_rows in gathered:
        if not rank_rows:
            continue
        for row in rank_rows:
            by_step.setdefault(int(row["step"]), []).append(row)

    merged_rows: List[Dict[str, Any]] = []
    metric_keys = [
        "em",
        "pr",
        "accuracy",
        "legal_rate",
        "hole_rate",
        "entropy",
        "normalized_entropy",
        "closer_target_rate",
    ]
    for step in sorted(by_step):
        chunks = by_step[step]
        sample_count = sum(int(chunk["samples"]) for chunk in chunks)
        if sample_count <= 0:
            raise ValueError(f"No samples gathered for checkpoint step {step}")
        merged = {
            "checkpoint": chunks[0]["checkpoint"],
            "step": step,
            "path": chunks[0]["path"],
            "samples": sample_count,
        }
        for key in metric_keys:
            merged[key] = sum(float(chunk[key]) * int(chunk["samples"]) for chunk in chunks) / sample_count
        merged_rows.append(merged)
    return merged_rows


def write_csv(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: List[Mapping[str, Any]]) -> str:
    headers = [
        "Checkpoint",
        "Step",
        "EM",
        "PR",
        "ACC",
        "Legal",
        "Hole",
        "Entropy",
        "Norm Entropy",
        "Closer+Target",
    ]
    keys = [
        "checkpoint",
        "step",
        "em",
        "pr",
        "accuracy",
        "legal_rate",
        "hole_rate",
        "entropy",
        "normalized_entropy",
        "closer_target_rate",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = []
        for key in keys:
            value = row[key]
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(
    rows: List[Mapping[str, Any]],
    output_path: Path,
    csv_path: Path,
    ckpt_root: Path,
    test_dataset: Path,
    task: str,
) -> None:
    best_em = max(rows, key=lambda row: float(row["em"]))
    best_pr = max(rows, key=lambda row: float(row["pr"]))
    best_entropy = max(rows, key=lambda row: float(row["normalized_entropy"]))
    relative_csv = csv_path
    try:
        relative_csv = csv_path.relative_to(output_path.parent)
    except ValueError:
        pass

    content = f"""# SSVR-T Evaluation

- Task: `{task}`
- Checkpoint root: `{ckpt_root}`
- Test dataset: `{test_dataset}`
- Samples: `{rows[0]["samples"] if rows else 0}`
- CSV: `{relative_csv}`

## Metrics

{markdown_table(rows)}

## Best checkpoints

- EM: `{best_em["checkpoint"]}`, EM={float(best_em["em"]):.4f}, PR={float(best_em["pr"]):.4f}
- PR: `{best_pr["checkpoint"]}`, EM={float(best_pr["em"]):.4f}, PR={float(best_pr["pr"]):.4f}
- Normalized entropy: `{best_entropy["checkpoint"]}`, value={float(best_entropy["normalized_entropy"]):.4f}
"""
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_root",
        default="output/models/frozenlake/UU_SFT_random_soft_ckpts",
    )
    parser.add_argument("--base_model", default="./models/LVM_ckpts")
    parser.add_argument("--backbone_type", default="auto", choices=("auto", "lvm", "qwen25vl"))
    parser.add_argument("--qwen_visual_input", action="store_true")
    parser.add_argument("--qwen_processor_path", default=None)
    parser.add_argument("--qwen_visual_image_size", type=int, default=256)
    parser.add_argument("--qwen_processor_use_fast", action="store_true")
    parser.add_argument("--qwen_visual_image_root", default=None)
    parser.add_argument("--qwen_prompt_text", default=None)
    parser.add_argument("--qwen_visual_image_scale", type=float, default=1.0)
    parser.add_argument("--use_kv_cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--test_dataset",
        default="dataset/frozenlake/tokenized_dataset/SFT/test_dataset.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="evaluation_results/frozenlake/UU_SFT_random_soft_ckpts_all_ckpt_eval",
    )
    parser.add_argument("--doc_path", default="frozenlake_sota.md")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--task", required=True, choices=("frozenlake", "maze", "minibehaviour", "maze_flip"))
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--torch_dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    args = parser.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    ckpt_root = Path(args.ckpt_root)
    test_dataset = Path(args.test_dataset)
    output_dir = Path(args.output_dir)
    doc_path = Path(args.doc_path)
    checkpoints = discover_checkpoints(ckpt_root)
    if not checkpoints:
        raise FileNotFoundError(f"No raw checkpoint-* directories found under {ckpt_root}")

    device = args.device or (f"cuda:{local_rank}" if distributed else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_dtype = parse_dtype(args.torch_dtype, device)
    samples = read_jsonl_samples(str(test_dataset), max_samples=args.max_samples)
    qwen_processor = None
    if args.qwen_visual_input:
        from transformers import AutoProcessor

        qwen_processor = AutoProcessor.from_pretrained(
            args.qwen_processor_path or args.base_model,
            trust_remote_code=True,
            use_fast=bool(args.qwen_processor_use_fast),
        )
    rank_samples = [sample for idx, sample in enumerate(samples) if idx % world_size == rank]

    rows: List[Dict[str, Any]] = []
    merged_progress_rows: List[Dict[str, Any]] = []
    for checkpoint in checkpoints:
        if rank == 0:
            print(f"Evaluating {checkpoint} on {len(samples)} samples across {world_size} rank(s)...")
        row = evaluate_one(
            checkpoint=checkpoint,
            base_model=args.base_model,
            backbone_type=args.backbone_type,
            samples=rank_samples,
            device=device,
            torch_dtype=torch_dtype,
            qwen_processor=qwen_processor,
            qwen_visual_image_size=int(args.qwen_visual_image_size),
            qwen_visual_image_root=args.qwen_visual_image_root,
            qwen_prompt_text_override=args.qwen_prompt_text,
            qwen_visual_image_scale=float(args.qwen_visual_image_scale),
            use_kv_cache=args.use_kv_cache,
        )
        rows.append(row)
        partial_rows = aggregate_rank_rows([row], rank, world_size)
        if rank == 0 and partial_rows:
            merged_progress_rows.extend(partial_rows)
            partial_path = output_dir / "summary_partial.json"
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(json.dumps(merged_progress_rows, indent=2, ensure_ascii=False), encoding="utf-8")
            merged_row = partial_rows[0]
            print(
                f"{checkpoint.name}: EM={merged_row['em']:.4f} PR={merged_row['pr']:.4f} "
                f"Legal={merged_row['legal_rate']:.4f} Entropy={merged_row['entropy']:.4f}"
            )

    merged_rows = aggregate_rank_rows(rows, rank, world_size)
    if distributed:
        dist.barrier()
    if rank != 0:
        if distributed:
            dist.destroy_process_group()
        return

    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    write_csv(merged_rows, csv_path)
    json_path.write_text(json.dumps(merged_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(
        rows=merged_rows,
        output_path=doc_path,
        csv_path=csv_path,
        ckpt_root=ckpt_root,
        test_dataset=test_dataset,
        task=args.task,
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {doc_path}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
