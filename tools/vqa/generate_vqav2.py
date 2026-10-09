#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.distributed as dist
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def init_distributed() -> Tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def sample_key(row: Mapping[str, Any]) -> str:
    return f"VQAv2::{int(row['question_id'])}"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def completed_keys(path: Path, require_native: bool = True) -> Set[str]:
    return {
        sample_key(row)
        for row in read_jsonl(path)
        if "model_answer" in row and (not require_native or "native_qwen_answer" in row)
    }


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_shards(
    shard_paths: Iterable[Path],
    allowed_keys: Optional[Set[str]] = None,
    require_native: bool = True,
) -> List[Dict[str, Any]]:
    dedup: Dict[str, Dict[str, Any]] = {}
    for shard_path in shard_paths:
        for row in read_jsonl(shard_path):
            if "model_answer" not in row or (require_native and "native_qwen_answer" not in row):
                continue
            if allowed_keys is not None and sample_key(row) not in allowed_keys:
                continue
            dedup[sample_key(row)] = row
    return sorted(dedup.values(), key=lambda row: int(row["sample_index"]))


def count_shard_rows(
    shard_paths: Iterable[Path],
    allowed_keys: Optional[Set[str]] = None,
    require_native: bool = True,
) -> int:
    return len(merge_shards(shard_paths, allowed_keys=allowed_keys, require_native=require_native))


def load_vqav2_rows(
    data_root: Path,
    max_samples: Optional[int],
    sample_seed: int,
    sample_strategy: str,
) -> List[Dict[str, Any]]:
    questions_path = data_root / "v2_OpenEnded_mscoco_val2014_questions.json"
    annotations_path = data_root / "v2_mscoco_val2014_annotations.json"
    image_dir = data_root / "val2014"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))["questions"]
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))["annotations"]
    annotation_by_qid = {int(row["question_id"]): row for row in annotations}

    rows: List[Dict[str, Any]] = []
    for idx, question in enumerate(questions):
        qid = int(question["question_id"])
        ann = annotation_by_qid.get(qid)
        if ann is None:
            continue
        image_id = int(question["image_id"])
        image_path = image_dir / f"COCO_val2014_{image_id:012d}.jpg"
        if not image_path.exists():
            continue
        human_answers = [str(answer["answer"]) for answer in ann.get("answers", [])]
        rows.append(
            {
                "dataset": "VQAv2_val",
                "sample_index": idx,
                "sample_id": qid,
                "question_id": qid,
                "image_id": image_id,
                "question": str(question["question"]),
                "question_type": ann.get("question_type"),
                "answer_type": ann.get("answer_type"),
                "gt_answer": str(ann.get("multiple_choice_answer", "")),
                "gt_answers": human_answers,
                "image_path": str(image_path),
            }
        )

    if max_samples is not None and int(max_samples) >= 0 and len(rows) > int(max_samples):
        max_samples = int(max_samples)
        if sample_strategy == "first":
            rows = rows[:max_samples]
        elif sample_strategy == "random":
            rng = random.Random(int(sample_seed))
            indices = sorted(rng.sample(range(len(rows)), max_samples))
            rows = [rows[idx] for idx in indices]
        else:
            raise ValueError(f"Unknown sample_strategy: {sample_strategy}")
    return rows


def resize_image_for_qwen(image: Image.Image, max_image_size: int) -> Tuple[Image.Image, Tuple[int, int]]:
    image = image.convert("RGB")
    original_size = image.size
    max_image_size = int(max_image_size)
    if max_image_size <= 0 or max(original_size) <= max_image_size:
        return image, original_size
    resized = image.copy()
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    resized.thumbnail((max_image_size, max_image_size), resampling)
    return resized, original_size


def load_model(base_model: str, checkpoint: str, dtype: torch.dtype, device: torch.device):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    adapter_dir = Path(checkpoint) / "backbone"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing trained backbone adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.to(device)
    model.eval()
    return model


def peft_adapter_disabled(model):
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return nullcontext()


def build_generation_messages(question_prompt: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question_prompt},
            ],
        }
    ]


@torch.no_grad()
def generate_outputs(
    model,
    processor,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    max_new_tokens: int,
    max_image_size: int,
    direct_answer_prompt: str,
    shard_path: Path,
    rank: int,
    progress_disable: bool,
    skip_native_qwen: bool = False,
) -> None:
    require_native = not bool(skip_native_qwen)
    done = completed_keys(shard_path, require_native=require_native)
    pending = [row for row in rows if sample_key(row) not in done]
    progress = None if progress_disable else tqdm(total=len(pending), desc=f"rank{rank} VQAv2")
    for row in pending:
        image_raw = Image.open(row["image_path"]).convert("RGB")
        image, original_size = resize_image_for_qwen(image_raw, max_image_size)
        prompt = str(row["question"]).strip()
        if direct_answer_prompt:
            prompt = f"{prompt}\n{direct_answer_prompt.strip()}"
        text = processor.apply_chat_template(
            build_generation_messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)

        def _generate_answer() -> str:
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
            return processor.batch_decode(
                completion,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

        native_qwen_answer = None
        if not skip_native_qwen:
            with peft_adapter_disabled(model):
                native_qwen_answer = _generate_answer()
        model_answer = _generate_answer()
        payload = {
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "sample_index": row["sample_index"],
                "question_id": row["question_id"],
                "image_id": row["image_id"],
                "question_type": row["question_type"],
                "answer_type": row["answer_type"],
                "question_prompt": prompt,
                "image_path": row["image_path"],
                "original_image_size": list(original_size),
                "model_image_size": list(image.size),
                "gt_answer": row["gt_answer"],
                "gt_answers": row["gt_answers"],
                "model_answer": model_answer,
            }
        if not skip_native_qwen:
            payload["native_qwen_answer"] = native_qwen_answer
        append_jsonl(shard_path, payload)
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()


def parse_max_samples(value: str) -> Optional[int]:
    value = str(value).strip().lower()
    if value in {"", "none", "null", "all", "-1"}:
        return None
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/usr/UU/data/VQAv2_val/downloads")
    parser.add_argument("--base_model", default="/home/usr/UU/m-x/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--trained_checkpoint", required=True)
    parser.add_argument("--processor_path", default=None)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--max_samples", default="1000")
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--sample_strategy", choices=("random", "first"), default="random")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_image_size", type=int, default=256)
    parser.add_argument("--direct_answer_prompt", default="Answer directly.")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", action="store_false", dest="resume")
    parser.add_argument("--torch_dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--processor_use_fast", action="store_true")
    parser.add_argument(
        "--skip_native_qwen",
        action="store_true",
        help="Do not generate the base/native Qwen answer; output model_answer only.",
    )
    args = parser.parse_args()

    distributed, rank, _local_rank, world_size, device = init_distributed()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]

    all_rows = load_vqav2_rows(
        data_root=Path(args.data_root),
        max_samples=parse_max_samples(args.max_samples),
        sample_seed=int(args.sample_seed),
        sample_strategy=str(args.sample_strategy),
    )
    target_keys = {sample_key(row) for row in all_rows}
    output_path = Path(args.output_json)
    shard_dir = output_path.parent / f"{output_path.stem}_shards"
    shard_paths = [shard_dir / f"rank{idx:05d}.jsonl" for idx in range(world_size)]
    shard_path = shard_paths[rank]
    if not args.resume and rank == 0:
        for path in shard_paths:
            if path.exists():
                path.unlink()
        if output_path.exists():
            output_path.unlink()
    if distributed:
        dist.barrier()

    rank_rows_all = [row for idx, row in enumerate(all_rows) if idx % world_size == rank]
    rank_done = (
        completed_keys(shard_path, require_native=not bool(args.skip_native_qwen))
        if args.resume
        else set()
    )
    rank_rows = [row for row in rank_rows_all if sample_key(row) not in rank_done]
    if rank == 0:
        print(
            f"VQAv2 rows: {len(all_rows)}; existing shard rows: "
            f"{count_shard_rows(shard_paths, allowed_keys=target_keys, require_native=not bool(args.skip_native_qwen))}; "
            f"resume={args.resume}"
        )

    processor = AutoProcessor.from_pretrained(
        args.processor_path or args.base_model,
        trust_remote_code=True,
        use_fast=bool(args.processor_use_fast),
    )
    model = load_model(args.base_model, args.trained_checkpoint, dtype=dtype, device=device)
    generate_outputs(
        model=model,
        processor=processor,
        rows=rank_rows,
        device=device,
        max_new_tokens=args.max_new_tokens,
        max_image_size=args.max_image_size,
        direct_answer_prompt=args.direct_answer_prompt,
        shard_path=shard_path,
        rank=rank,
        progress_disable=rank != 0,
        skip_native_qwen=bool(args.skip_native_qwen),
    )

    if distributed:
        dist.barrier()
    if rank == 0:
        outputs = merge_shards(
            shard_paths,
            allowed_keys=target_keys,
            require_native=not bool(args.skip_native_qwen),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {len(outputs)} VQAv2 model outputs to {output_path}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
