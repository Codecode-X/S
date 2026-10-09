import argparse
import sys
import torch
import torch.distributed as dist
import json
import jsonlines
import logging
import math
import random
import numpy as np
import os
from pathlib import Path
from peft import PeftModel
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_callback import ProgressCallback
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence
import swanlab
SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ssvr.backbone import load_backbone_model
from ssvr.logging_utils import (
    find_swanlab_resume_id,
    make_swanlab_callback,
    make_run_id_saver,
    resolve_swanlab_project,
)
from ssvr.evaluation.evaluator import run_local_action_inference_on_sample
from ssvr.evaluation.metrics import compute_em_pr, summarize_local_action_results
from ssvr.reasoning import (
    LocalActionPlanningModel,
    QwenVisualLocalActionCollator,
    SFTRandomLocalActionDataset,
    local_action_valid_mask_collate_fn,
    load_local_action_components,
    action_names_for_meta,
    normalize_state,
    qwen_visual_prompt_text,
    qwen_visual_image_from_tokens,
    qwen_visual_tokens_for_sample,
    build_qwen_visual_start_tokens_index,
    model_action_history,
    ordered_step_index,
    save_local_action_model,
    state_carrying,
    state_to_task_position,
    task_distance_to_goal,
    transition_step,
    valid_action_mask,
    is_action_legal,
    MINIBEHAVIOUR_ACTION_TO_ID,
    is_maze_flip_meta,
    maze_flip_view_action_to_world,
    maze_flip_world_mask_to_view,
)

log = logging.getLogger(__name__)


def log_stage(message: str) -> None:

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        log.info("========== [STAGE] %s ==========" , message)

EXPECTED_SSVR_T_TRAIN_SAMPLES = {
    "frozenlake": 162425,
    "maze": 156682,
    "minibehaviour": 90808,
}

dtype_map = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16
}


class QuietProgressCallback(ProgressCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):
        return control


def install_quiet_progress_callback(trainer: Trainer) -> None:
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(QuietProgressCallback)

def seed_everything(seed = 42):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda_backend(tf32: bool) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
    precision = "high" if tf32 else "highest"
    torch.set_float32_matmul_precision(precision)

class StopAfterEpochCallback(TrainerCallback):

    def __init__(self, stop_after_epochs: float):
        self.stop_after_epochs = float(stop_after_epochs)
        if self.stop_after_epochs <= 0:
            raise ValueError("stop_after_epochs must be positive")

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is not None and float(state.epoch) >= self.stop_after_epochs:
            control.should_training_stop = True
        return control


def entropy_from_probs(probs: Sequence[float]) -> float:
    return -sum(float(p) * math.log(float(p)) for p in probs if float(p) > 0.0)


def read_jsonl_eval_samples(path: str, max_samples: Optional[int] = None) -> List[Mapping[str, Any]]:
    samples: List[Mapping[str, Any]] = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            samples.append(obj)
            if max_samples is not None and len(samples) >= int(max_samples):
                break
    return samples


def unwrap_local_action_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def build_qwen_visual_eval_tensors(
    sample: Mapping[str, Any],
    processor: Any,
    device: torch.device,
    image_size: int = 256,
    image_root: Optional[str] = None,
    start_tokens_index: Optional[Mapping[Any, Any]] = None,
    prompt_text_override: Optional[str] = None,
    image_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    meta = sample["meta"]
    current_state = int(sample.get("current_state", normalize_state(sample["input_state"], meta)))
    prompt_text = prompt_text_override or qwen_visual_prompt_text(meta, current_state)
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
    if not image_root:
        raise ValueError(
            "Qwen native visual evaluation requires qwen_visual_image_root with "
            "VQ-decoded initial-map images."
        )
    image = qwen_visual_image_from_tokens(
        image_root,
        qwen_visual_tokens_for_sample(sample, start_tokens_index=start_tokens_index),
        scale=image_scale,
    )
    qwen_inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    return {
        "qwen_input_ids": qwen_inputs["input_ids"].to(device),
        "qwen_attention_mask": qwen_inputs["attention_mask"].to(device),
        "qwen_pixel_values": qwen_inputs["pixel_values"].to(device),
        "qwen_image_grid_thw": qwen_inputs["image_grid_thw"].to(device),
    }


@torch.no_grad()
def compute_stage1_local_action_metrics(
    model: torch.nn.Module,
    samples: Sequence[Mapping[str, Any]],
    device: Optional[torch.device] = None,
    use_cache: bool = True,
    qwen_processor: Optional[Any] = None,
    qwen_visual_image_size: int = 256,
    qwen_visual_image_root: Optional[str] = None,
    qwen_prompt_text_override: Optional[str] = None,
    qwen_visual_image_scale: float = 1.0,
) -> Dict[str, float]:

    unwrapped = unwrap_local_action_model(model)
    if device is None:
        device = next(unwrapped.parameters()).device
    was_training = unwrapped.training
    unwrapped.eval()
    qwen_start_tokens_index = (
        build_qwen_visual_start_tokens_index(samples)
        if qwen_processor is not None and qwen_visual_image_root
        else None
    )

    totals = {
        "legal": 0.0,
        "hole": 0.0,
        "entropy": 0.0,
        "normalized_entropy": 0.0,
        "closer_target": 0.0,
        "pick": 0.0,
        "drop": 0.0,
        "pick_state": 0.0,
        "pick_state_pick": 0.0,
        "drop_state": 0.0,
        "drop_state_drop": 0.0,
    }
    count = 0
    step_counters: Dict[Any, int] = {}
    try:
        for sample in samples:
            meta = sample["meta"]
            state = normalize_state(sample["input_state"], meta)
            state_step_value = ordered_step_index(step_counters, meta, state)
            height = int(meta.get("height", meta.get("level")))
            width = int(meta.get("width", meta.get("level")))
            row, col = state_to_task_position(meta, state)

            global_input_ids = torch.tensor(
                sample["input_tokens"],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            position = torch.tensor([[row, col]], dtype=torch.float32, device=device)
            map_size = torch.tensor([[height, width]], dtype=torch.float32, device=device)
            state_flags = torch.tensor(
                [[float(state_carrying(meta, state))]],
                dtype=torch.float32,
                device=device,
            )
            state_step = torch.tensor([state_step_value], dtype=torch.float32, device=device)
            if is_maze_flip_meta(meta):
                position = torch.zeros_like(position)
                map_size = torch.ones_like(map_size)
                state_flags = torch.zeros_like(state_flags)
                state_step = torch.zeros_like(state_step)
            action_history_values = model_action_history(meta, state)
            state_action_history = torch.tensor(
                [action_history_values], dtype=torch.long, device=device
            )
            state_action_history_mask = torch.ones_like(
                state_action_history, dtype=torch.bool
            )
            qwen_visual_tensors = (
                build_qwen_visual_eval_tensors(
                    sample,
                    qwen_processor,
                    device,
                    image_size=qwen_visual_image_size,
                    image_root=qwen_visual_image_root,
                    start_tokens_index=qwen_start_tokens_index,
                    prompt_text_override=qwen_prompt_text_override,
                    image_scale=qwen_visual_image_scale,
                )
                if qwen_processor is not None
                else {}
            )

            _, logits = unwrapped.predict_action(
                global_input_ids=global_input_ids,
                position=position,
                map_size=map_size,
                use_cache=use_cache,
                state_flags=state_flags,
                state_step=state_step,
                state_action_history=state_action_history,
                state_action_history_mask=state_action_history_mask,
                **qwen_visual_tensors,
            )
            action_count = len(action_names_for_meta(meta))
            probs = torch.softmax(logits.float(), dim=-1)[0, :action_count].detach().cpu()
            action_id = int(torch.argmax(probs).item())
            world_action_id = maze_flip_view_action_to_world(meta, state, action_id)
            result = transition_step(meta, state, world_action_id)
            action_names = action_names_for_meta(meta)

            world_legal_mask = valid_action_mask(meta, state)
            model_legal_mask = maze_flip_world_mask_to_view(meta, state, world_legal_mask)
            legal_ids = [
                action_idx
                for action_idx, is_valid in enumerate(model_legal_mask)
                if float(is_valid) > 0.0
            ]
            legal_probs = probs[legal_ids] if legal_ids else torch.empty(0)
            legal_mass = float(legal_probs.sum().item()) if legal_ids else 0.0
            if legal_ids and legal_mass > 0.0:
                conditional = (legal_probs / legal_mass).tolist()
                entropy = entropy_from_probs(conditional)
                normalized_entropy = (
                    entropy / math.log(len(legal_ids)) if len(legal_ids) > 1 else 0.0
                )
            else:
                entropy = 0.0
                normalized_entropy = 0.0

            closer_or_target = False
            if result.valid:
                if result.reason in {"target", "pick"}:
                    closer_or_target = True
                else:
                    current_distance = task_distance_to_goal(meta, state)
                    next_distance = task_distance_to_goal(meta, result.next_state)
                    closer_or_target = (
                        current_distance is not None
                        and next_distance is not None
                        and float(next_distance) == float(current_distance) - 1.0
                    )

            totals["legal"] += 1.0 if result.valid else 0.0
            totals["hole"] += 1.0 if result.reason == "hole" else 0.0
            totals["entropy"] += entropy
            totals["normalized_entropy"] += normalized_entropy
            totals["closer_target"] += 1.0 if closer_or_target else 0.0
            if len(action_names) > 4:
                pick_id = MINIBEHAVIOUR_ACTION_TO_ID["PICK"]
                drop_id = MINIBEHAVIOUR_ACTION_TO_ID["DROP"]
                totals["pick"] += 1.0 if action_id == pick_id and result.valid else 0.0
                totals["drop"] += 1.0 if action_id == drop_id and result.valid else 0.0
                if is_action_legal(meta, state, pick_id):
                    totals["pick_state"] += 1.0
                    totals["pick_state_pick"] += 1.0 if action_id == pick_id else 0.0
                if is_action_legal(meta, state, drop_id):
                    totals["drop_state"] += 1.0
                    totals["drop_state_drop"] += 1.0 if action_id == drop_id else 0.0
            count += 1
    finally:
        if was_training:
            unwrapped.train()

    if count == 0:
        return {
            "legal_rate": 0.0,
            "hole_rate": 0.0,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "closer_target_rate": 0.0,
            "eval_samples": 0.0,
        }

    return {
        "legal_rate": totals["legal"] / count,
        "hole_rate": totals["hole"] / count,
        "entropy": totals["entropy"] / count,
        "normalized_entropy": totals["normalized_entropy"] / count,
        "closer_target_rate": totals["closer_target"] / count,
        "pick_rate": totals["pick"] / count,
        "drop_rate": totals["drop"] / count,
        "pick_legal_state_pick_rate": (
            totals["pick_state_pick"] / totals["pick_state"]
            if totals["pick_state"] > 0.0
            else 0.0
        ),
        "drop_legal_state_drop_rate": (
            totals["drop_state_drop"] / totals["drop_state"]
            if totals["drop_state"] > 0.0
            else 0.0
        ),
        "pick_legal_state_count": totals["pick_state"],
        "drop_legal_state_count": totals["drop_state"],
        "eval_samples": float(count),
    }


@torch.no_grad()
def compute_stage1_closed_loop_metrics(
    model: torch.nn.Module,
    samples: Sequence[Mapping[str, Any]],
    device: Optional[torch.device] = None,
    use_cache: bool = True,
    qwen_processor: Optional[Any] = None,
    qwen_visual_image_size: int = 256,
    qwen_visual_image_root: Optional[str] = None,
    qwen_prompt_text_override: Optional[str] = None,
    qwen_visual_image_scale: float = 1.0,
) -> Dict[str, float]:

    device = device or next(model.parameters()).device
    unwrapped = unwrap_local_action_model(model)
    was_training = unwrapped.training
    unwrapped.eval()
    results: List[Dict[str, Any]] = []
    try:
        for sample_idx, sample in enumerate(samples):
            meta = sample["meta"]
            mini = len(action_names_for_meta(meta)) > 4
            result = run_local_action_inference_on_sample(
                model=unwrapped,
                sample=sample,
                device=device,
                task="minibehaviour" if mini else "local_action",
                use_cache=use_cache,
                qwen_visual_input=qwen_processor is not None,
                qwen_processor=qwen_processor,
                qwen_visual_image_size=qwen_visual_image_size,
                qwen_visual_image_root=qwen_visual_image_root,
                qwen_prompt_text_override=qwen_prompt_text_override,
                qwen_visual_image_scale=qwen_visual_image_scale,
            )

            level = int(meta["level"])
            width = int(meta.get("width", level))
            if mini:
                em, pr = float(result["em"]), float(result["pr"])
            else:
                em, pr = compute_em_pr(
                    start_coords=result["start_coords"],
                    expected_move=int(result["expected_move"]),
                    distance_map=result["distance_map"],
                    target_pos=int(result["target_pos"]),
                    width=width,
                )

            results.append(
                {
                    **result,
                    "sample_index": sample_idx,
                    "em": em,
                    "pr": pr,
                    "accuracy": 1.0 if result["complete"] else 0.0,
                    "width": width,
                    "height": int(meta.get("height", level)),
                }
            )
    finally:
        if was_training:
            unwrapped.train()

    summary = summarize_local_action_results(results)
    overall = summary["overall"]
    return {
        "closed_loop_em": float(overall["em"]),
        "closed_loop_pr": float(overall["pr"]),
        "closed_loop_accuracy": float(overall["accuracy"]),
        "closed_loop_samples": float(overall["count"]),
    }


class SFTStage1EpochMetricsCallback(TrainerCallback):

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        use_cache: bool = True,
        output_dir: Optional[str] = None,
        qwen_visual_input: bool = False,
        qwen_processor_path: Optional[str] = None,
        qwen_visual_image_size: int = 256,
        qwen_processor_use_fast: bool = False,
        qwen_visual_image_root: Optional[str] = None,
    ):
        self.samples = list(samples)
        self.use_cache = bool(use_cache)
        self.output_dir = output_dir
        self.qwen_visual_input = bool(qwen_visual_input)
        self.qwen_processor_path = qwen_processor_path
        self.qwen_visual_image_size = int(qwen_visual_image_size)
        self.qwen_processor_use_fast = bool(qwen_processor_use_fast)
        self.qwen_visual_image_root = (
            None if qwen_visual_image_root in (None, "") else str(qwen_visual_image_root)
        )
        self._qwen_processor = None

    @staticmethod
    def _weighted_average(
        rows: Sequence[Mapping[str, float]],
        key: str,
        weight_key: str,
    ) -> float:
        denominator = sum(float(row.get(weight_key, 0.0)) for row in rows)
        if denominator <= 0.0:
            return 0.0
        return (
            sum(float(row.get(key, 0.0)) * float(row.get(weight_key, 0.0)) for row in rows)
            / denominator
        )

    @classmethod
    def _merge_metric_rows(
        cls,
        local_metrics: Sequence[Mapping[str, float]],
        local_closed_loop_metrics: Sequence[Mapping[str, float]],
    ) -> Dict[str, float]:
        eval_samples = sum(float(row.get("eval_samples", 0.0)) for row in local_metrics)
        closed_loop_samples = sum(
            float(row.get("closed_loop_samples", 0.0)) for row in local_closed_loop_metrics
        )
        pick_legal_state_count = sum(
            float(row.get("pick_legal_state_count", 0.0)) for row in local_metrics
        )
        drop_legal_state_count = sum(
            float(row.get("drop_legal_state_count", 0.0)) for row in local_metrics
        )
        return {
            "sft_random_eval/legal_rate": cls._weighted_average(
                local_metrics, "legal_rate", "eval_samples"
            ),
            "sft_random_eval/hole_rate": cls._weighted_average(
                local_metrics, "hole_rate", "eval_samples"
            ),
            "sft_random_eval/entropy": cls._weighted_average(
                local_metrics, "entropy", "eval_samples"
            ),
            "sft_random_eval/normalized_entropy": cls._weighted_average(
                local_metrics, "normalized_entropy", "eval_samples"
            ),
            "sft_random_eval/closer_target_rate": cls._weighted_average(
                local_metrics, "closer_target_rate", "eval_samples"
            ),
            "sft_random_eval/em": cls._weighted_average(
                local_closed_loop_metrics, "closed_loop_em", "closed_loop_samples"
            ),
            "sft_random_eval/pr": cls._weighted_average(
                local_closed_loop_metrics, "closed_loop_pr", "closed_loop_samples"
            ),
            "sft_random_eval/accuracy": cls._weighted_average(
                local_closed_loop_metrics, "closed_loop_accuracy", "closed_loop_samples"
            ),
            "sft_random_eval/pick_rate": cls._weighted_average(
                local_metrics, "pick_rate", "eval_samples"
            ),
            "sft_random_eval/drop_rate": cls._weighted_average(
                local_metrics, "drop_rate", "eval_samples"
            ),
            "sft_random_eval/pick_legal_state_pick_rate": (
                sum(
                    float(row.get("pick_legal_state_pick_rate", 0.0))
                    * float(row.get("pick_legal_state_count", 0.0))
                    for row in local_metrics
                )
                / pick_legal_state_count
                if pick_legal_state_count > 0.0
                else 0.0
            ),
            "sft_random_eval/drop_legal_state_drop_rate": (
                sum(
                    float(row.get("drop_legal_state_drop_rate", 0.0))
                    * float(row.get("drop_legal_state_count", 0.0))
                    for row in local_metrics
                )
                / drop_legal_state_count
                if drop_legal_state_count > 0.0
                else 0.0
            ),
            "sft_random_eval/pick_legal_state_count": pick_legal_state_count,
            "sft_random_eval/drop_legal_state_count": drop_legal_state_count,
            "sft_random_eval/eval_samples": eval_samples,
            "sft_random_eval/closed_loop_samples": closed_loop_samples,
        }

    def on_epoch_end(self, args, state, control, **kwargs):
        if dist.is_initialized():
            dist.barrier()

        model = kwargs.get("model")
        if model is None:
            raise ValueError("SFTStage1EpochMetricsCallback requires Trainer to pass model")
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        local_samples = [
            sample for sample_idx, sample in enumerate(self.samples) if sample_idx % world_size == rank
        ]
        local_metrics: Dict[str, float]
        local_closed_loop_metrics: Dict[str, float]
        if local_samples:
            unwrapped = unwrap_local_action_model(model)
            device = next(unwrapped.parameters()).device
            if self.qwen_visual_input and self._qwen_processor is None:
                from transformers import AutoProcessor

                self._qwen_processor = AutoProcessor.from_pretrained(
                    self.qwen_processor_path,
                    trust_remote_code=True,
                    use_fast=self.qwen_processor_use_fast,
                )
            local_metrics = compute_stage1_local_action_metrics(
                model=unwrapped,
                samples=local_samples,
                device=device,
                use_cache=self.use_cache,
                qwen_processor=self._qwen_processor,
                qwen_visual_image_size=self.qwen_visual_image_size,
                qwen_visual_image_root=self.qwen_visual_image_root,
            )
            local_closed_loop_metrics = compute_stage1_closed_loop_metrics(
                model=unwrapped,
                samples=local_samples,
                device=device,
                use_cache=self.use_cache,
                qwen_processor=self._qwen_processor,
                qwen_visual_image_size=self.qwen_visual_image_size,
                qwen_visual_image_root=self.qwen_visual_image_root,
            )
        else:
            local_metrics = compute_stage1_local_action_metrics(
                model=unwrap_local_action_model(model),
                samples=[],
            )
            local_closed_loop_metrics = {
                "closed_loop_em": 0.0,
                "closed_loop_pr": 0.0,
                "closed_loop_accuracy": 0.0,
                "closed_loop_samples": 0.0,
            }

        if dist.is_initialized():
            gathered_metrics: List[Dict[str, float]] = [None for _ in range(world_size)]
            gathered_closed: List[Dict[str, float]] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_metrics, local_metrics)
            dist.all_gather_object(gathered_closed, local_closed_loop_metrics)
        else:
            gathered_metrics = [local_metrics]
            gathered_closed = [local_closed_loop_metrics]

        if rank == 0 and self.samples:
            log_payload = {
                **self._merge_metric_rows(gathered_metrics, gathered_closed),
                "train/epoch": float(state.epoch or 0.0),
                "train/global_step": int(state.global_step),
            }
            if swanlab.has_run():
                try:
                    swanlab.log(log_payload)
                except Exception as exc:
                    log.warning("Stage-1 SwanLab metric logging failed: %s", exc)
            if self.output_dir:
                metrics_path = Path(self.output_dir) / "stage1_epoch_metrics.jsonl"
                try:
                    metrics_path.parent.mkdir(parents=True, exist_ok=True)
                    with metrics_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(log_payload, ensure_ascii=False) + "\n")
                except Exception as exc:
                    log.warning("Stage-1 metric file write failed: %s", exc)
            log.info("Stage-1 epoch metrics: %s", log_payload)

        if dist.is_initialized():
            dist.barrier()
        return control


def freeze_unused_qwen25vl_visual_lora(model):
    frozen = 0
    for name, param in model.named_parameters():
        if ".visual." in name and ".lora_" in name and param.requires_grad:
            param.requires_grad_(False)
            frozen += int(param.numel())
    if frozen:
        log.info("Froze %s Qwen2.5-VL visual-tower LoRA parameters for local-action training", frozen)
    return model


def create_lora_model(model, freeze_qwen_visual_lora: bool = True):
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj"
        ]
    )
    model = get_peft_model(model, lora_config)
    if freeze_qwen_visual_lora:
        model = freeze_unused_qwen25vl_visual_lora(model)
    return model


def maybe_qwen_visual_sft_collator(
    cfg: Any,
    base_collator,
    include_valid_action_mask: bool = False,
):
    if not bool(getattr(cfg.SFT, "qwen_visual_input", False)):
        return base_collator
    processor_path = str(getattr(cfg.SFT, "qwen_processor_path", None) or cfg.SFT.model_path)
    log.info("Using Qwen native visual inputs with processor: %s", processor_path)
    return QwenVisualLocalActionCollator(
        processor_path=processor_path,
        include_valid_action_mask=include_valid_action_mask,
        image_size=int(getattr(cfg.SFT, "qwen_visual_image_size", 256)),
        processor_use_fast=bool(getattr(cfg.SFT, "qwen_processor_use_fast", False)),
        image_root=getattr(cfg.SFT, "qwen_visual_image_root", None),
    )


def _parse_action_text_labels(
    raw_labels: Optional[Any],
    action_names: Sequence[str],
) -> List[str]:
    if raw_labels is None:
        return [str(name) for name in action_names]
    if isinstance(raw_labels, str):
        stripped = raw_labels.strip()
        if not stripped or stripped.lower() in {"null", "none"}:
            return [str(name) for name in action_names]
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return [str(item).strip() for item in raw_labels if str(item).strip()]


def resolve_action_token_ids_for_sft(
    cfg: Any,
    action_names: Sequence[str],
) -> Optional[List[int]]:
    labels = _parse_action_text_labels(getattr(cfg.SFT, "action_text_labels", None), action_names)
    if len(labels) != len(action_names):
        raise ValueError(
            "SFT.action_text_labels must provide exactly one text label per action: "
            f"{len(labels)} != {len(action_names)}"
        )

    from transformers import AutoTokenizer

    tokenizer_path = str(getattr(cfg.SFT, "qwen_processor_path", None) or cfg.SFT.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=bool(getattr(cfg.SFT, "qwen_processor_use_fast", False)),
    )
    token_ids: List[int] = []
    for action_name, label in zip(action_names, labels):
        ids = tokenizer.encode(label, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Action text label for {action_name!r} produced no tokens: {label!r}")
        if len(ids) > 1:
            log.warning(
                "SFT.action_output_mode=text_token uses only the first token for action %s: "
                "label=%r token_ids=%s",
                action_name,
                label,
                ids,
            )
        token_ids.append(int(ids[0]))
    log.info(
        "Text-token action output labels: %s",
        {name: {"label": label, "token_id": token_id} for name, label, token_id in zip(action_names, labels, token_ids)},
    )
    return token_ids


def create_sft_training_args(cfg: Any, bf16: bool = True) -> TrainingArguments:
    num_workers = int(cfg.SFT.dataloader_num_workers)
    return TrainingArguments(
        output_dir=cfg.SFT.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=cfg.SFT.num_train_epochs,
        per_device_train_batch_size=cfg.SFT.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.SFT.gradient_accumulation_steps,
        learning_rate=float(cfg.SFT.learning_rate),
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=bf16,
        save_strategy="epoch",
        save_total_limit=20,
        logging_strategy="steps",
        logging_steps=1,
        report_to="none",
        optim=cfg.SFT.optim,
        tf32=cfg.SFT.tf32,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=cfg.SFT.dataloader_pin_memory,
        dataloader_persistent_workers=cfg.SFT.dataloader_persistent_workers and num_workers > 0,
        ddp_find_unused_parameters=False,
        run_name=cfg.SFT.run_name,
        remove_unused_columns=False,
    )


def create_sft_callbacks(
    cfg: Any,
    project: str,
    resume_ckpt: Optional[str],
    stage1_eval_samples: Optional[Sequence[Mapping[str, Any]]] = None,
):
    from accelerate import PartialState

    is_main_process = PartialState().is_main_process
    callbacks = []

    class DDPSyncCallback(TrainerCallback):

        def on_log(self, args, state, control, **kwargs):
            if dist.is_initialized():
                dist.barrier()

        def on_save(self, args, state, control, **kwargs):
            if dist.is_initialized():
                dist.barrier()

    stop_after_epochs = getattr(cfg.SFT, "stop_after_epochs", None)
    if stop_after_epochs is not None:
        stop_after_epochs = float(stop_after_epochs)
        scheduled_epochs = float(cfg.SFT.num_train_epochs)
        if stop_after_epochs > scheduled_epochs:
            raise ValueError(
                "SFT.stop_after_epochs cannot exceed SFT.num_train_epochs: "
                f"{stop_after_epochs} > {scheduled_epochs}"
            )
        callbacks.append(StopAfterEpochCallback(stop_after_epochs))
        log.info(
            "SFT will stop after epoch %s while retaining the %s-epoch LR schedule",
            stop_after_epochs,
            scheduled_epochs,
        )

    swanlab_resume_id: Optional[str] = None
    swanlab_resume_mode = "never"
    if resume_ckpt:
        log.info(f"Resuming training from checkpoint: {resume_ckpt}")
        swanlab_resume_id, swanlab_resume_mode = find_swanlab_resume_id(
            run_name=cfg.SFT.run_name,
            output_dir=cfg.SFT.output_dir,
        )
        if swanlab_resume_id:
            log.info(
                f"SwanLab resume: continuing run_id={swanlab_resume_id} "
                f"(mode={swanlab_resume_mode}) for {cfg.SFT.run_name}"
            )

    swanlab_mode = os.environ.get("SWANLAB_MODE", "cloud").strip().lower()
    swanlab_disabled = swanlab_mode in {
        "0",
        "false",
        "none",
        "off",
        "disable",
        "disabled",
    }

    if is_main_process and not swanlab_disabled:
        resolved_project = resolve_swanlab_project(project, "SWANLAB_SFT_PROJECT")
        log.info("SwanLab project: %s", resolved_project)
        swanlab_callback = make_swanlab_callback(
            project=resolved_project,
            experiment_name=cfg.SFT.run_name,
            mode=swanlab_mode,
        )
        if swanlab_resume_id:
            swanlab_callback._init_kwargs["id"] = swanlab_resume_id
            swanlab_callback._init_kwargs["resume"] = swanlab_resume_mode
        callbacks.append(swanlab_callback)
        callbacks.append(make_run_id_saver(cfg.SFT.output_dir))
    elif is_main_process:
        log.info("SwanLab callback disabled by SWANLAB_MODE=%r", swanlab_mode)

    if stage1_eval_samples:
        callbacks.append(
            SFTStage1EpochMetricsCallback(
                stage1_eval_samples,
                output_dir=str(cfg.SFT.output_dir),
                use_cache=os.environ.get("USE_KV_CACHE", "0") == "1",
                qwen_visual_input=bool(getattr(cfg.SFT, "qwen_visual_input", False)),
                qwen_processor_path=str(
                    getattr(cfg.SFT, "qwen_processor_path", None) or cfg.SFT.model_path
                ),
                qwen_visual_image_size=int(getattr(cfg.SFT, "qwen_visual_image_size", 256)),
                qwen_processor_use_fast=bool(
                    getattr(cfg.SFT, "qwen_processor_use_fast", False)
                ),
                qwen_visual_image_root=getattr(cfg.SFT, "qwen_visual_image_root", None),
            )
        )
        log.info("Stage-1 epoch metrics enabled on %s samples", len(stage1_eval_samples))

    callbacks.append(DDPSyncCallback())
    return callbacks, is_main_process


class LocalActionTrainer(Trainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        del _internal_call
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if not self.args.should_save:
            return

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        save_local_action_model(unwrapped_model, output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        model = model if model is not None else self.model
        unwrapped_model = self.accelerator.unwrap_model(model)
        load_local_action_components(unwrapped_model, resume_from_checkpoint)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        metrics = getattr(unwrapped_model, "_last_loss_metrics", None)
        if metrics:
            logs = {
                **logs,
                **{
                    f"sft_random_train/{key}": float(value)
                    for key, value in metrics.items()
                },
            }
        return super().log(logs, *args, **kwargs)


def train_local_action_sft_random(cfg: Any, torch_dtype: torch.dtype):
    log_stage("1/9 Validate training configuration and distributed batch size")
    resume_ckpt = os.environ.get("RESUME_FROM_CHECKPOINT", None)
    if not cfg.SFT.use_lora and not cfg.SFT.freeze_backbone:
        raise ValueError(
            "local_action SFT-Random refuses full-backbone finetuning. "
            "Set SFT.use_lora=True or SFT.freeze_backbone=True."
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = (
        int(cfg.SFT.per_device_train_batch_size)
        * int(cfg.SFT.gradient_accumulation_steps)
        * world_size
    )
    log.info(
        "Local-action SFT-Random effective global batch: %s "
        "(per_device=%s, grad_accum=%s, world_size=%s)",
        global_batch,
        cfg.SFT.per_device_train_batch_size,
        cfg.SFT.gradient_accumulation_steps,
        world_size,
    )

    sft_random_loss = str(cfg.SFT.sft_random_loss)
    if sft_random_loss != "optimal_biased_target_closer_not_closer":
        raise ValueError(
            "SSVR-T reproduction requires "
            "SFT.sft_random_loss=optimal_biased_target_closer_not_closer"
        )
    log_stage("2/9 Load and validate the SFT-Random training dataset")
    dataset = SFTRandomLocalActionDataset(
        cfg.SFT.dataset_pth,
        require_action_label="minibehaviour" not in str(cfg.SFT.dataset_pth).lower(),
        optimal_bias_alpha=float(cfg.SFT.sft_random_optimal_bias_alpha),
    )
    dataset_path_lower = str(cfg.SFT.dataset_pth).lower()
    dataset_task = next(
        (task for task in EXPECTED_SSVR_T_TRAIN_SAMPLES if task in dataset_path_lower),
        None,
    )
    if dataset_task is None:
        raise ValueError(
            f"Cannot identify SSVR-T task from dataset path: {cfg.SFT.dataset_pth}"
        )
    expected_samples = EXPECTED_SSVR_T_TRAIN_SAMPLES[dataset_task]
    if len(dataset) > expected_samples:
        raise ValueError(
            f"SSVR-T {dataset_task} sample count exceeds source dataset: "
            f"got {len(dataset)}, expected at most {expected_samples}"
        )
    log.info(
        "Dynamic-state reachable sample count verified: task=%s, samples=%s, source=%s",
        dataset_task,
        len(dataset),
        expected_samples,
    )
    steps_per_epoch = math.ceil(len(dataset) / global_batch)
    scheduler_steps = math.ceil(float(cfg.SFT.num_train_epochs) * steps_per_epoch)
    warmup_steps = math.ceil(0.1 * scheduler_steps)
    log.info(
        "SSVR-T scheduler plan: steps_per_epoch=%s, total_steps=%s, "
        "warmup_steps=%s, peak_lr=%.8g, scheduler=cosine",
        steps_per_epoch,
        scheduler_steps,
        warmup_steps,
        float(cfg.SFT.learning_rate),
    )
    action_names = dataset.action_names


    log_stage("3/9 Load the Qwen2.5-VL backbone")
    backbone = load_backbone_model(
        cfg.SFT.model_path,
        torch_dtype=torch_dtype,
        backbone_type=getattr(cfg.SFT, "backbone_type", "auto"),
    )

    if cfg.SFT.use_lora:
        log_stage("4/9 Create or restore the LoRA adapter")
        adapter_dir = Path(resume_ckpt) / "backbone" if resume_ckpt else None
        if adapter_dir and adapter_dir.exists():
            log.info(f"Loading local-action LoRA adapter from {adapter_dir}")
            backbone = PeftModel.from_pretrained(
                backbone,
                str(adapter_dir),
                is_trainable=True,
            )
            if not bool(getattr(cfg.SFT, "qwen_visual_input", False)):
                backbone = freeze_unused_qwen25vl_visual_lora(backbone)
        else:
            log.info("Using LoRA for local-action SFT-Random")
            backbone = create_lora_model(
                backbone,
                freeze_qwen_visual_lora=not bool(getattr(cfg.SFT, "qwen_visual_input", False)),
            )
        backbone.print_trainable_parameters()

    log_stage("5/9 Build the implicit-state token and action-output model")
    action_token_ids = resolve_action_token_ids_for_sft(cfg, action_names)
    model = LocalActionPlanningModel(
        backbone=backbone,
        num_frequencies=cfg.SFT.position_num_frequencies,
        position_mlp_hidden_size=cfg.SFT.position_mlp_hidden_size,
        global_start_token_id=cfg.SFT.global_start_token_id,
        global_end_token_id=cfg.SFT.global_end_token_id,
        action_names=action_names,
        action_token_ids=action_token_ids,
        action_output_mode=cfg.SFT.action_output_mode,
        ablation_modules=getattr(cfg.SFT, "ablation_modules", False),
    )
    if model.state_flag_fusion_mode != str(cfg.SFT.state_flag_fusion_mode):
        raise ValueError("SSVR-T reproduction requires state_flag_fusion_mode=add")
    if model.state_token_mode != str(cfg.SFT.state_token_mode):
        raise ValueError("SSVR-T reproduction requires state_token_mode=append")
    if model.action_output_mode != str(cfg.SFT.action_output_mode):
        raise ValueError("Model action output mode does not match training configuration")
    log.info("Local-action state token mode: %s", model.state_token_mode)
    log.info("Local-action output mode: %s", model.action_output_mode)
    log.info(f"SFT-Random dataset loaded: {len(dataset)} samples from {cfg.SFT.dataset_pth}")
    data_collator = maybe_qwen_visual_sft_collator(
        cfg,
        local_action_valid_mask_collate_fn,
        include_valid_action_mask=True,
    )


    log_stage("6/9 Build the collator, training arguments and evaluation callbacks")
    training_args = create_sft_training_args(cfg)
    stage1_eval_samples = None
    eval_dataset_pth = getattr(cfg.SFT, "eval_dataset_pth", None)
    if eval_dataset_pth:
        eval_max_samples = getattr(cfg.SFT, "eval_max_samples", None)
        stage1_eval_samples = read_jsonl_eval_samples(
            str(eval_dataset_pth),
            None if eval_max_samples is None else int(eval_max_samples),
        )
        log.info(
            "Loaded %s Stage-1 eval samples from %s",
            len(stage1_eval_samples),
            eval_dataset_pth,
        )
    callbacks, is_main_process = create_sft_callbacks(
        cfg=cfg,
        project="SFT_local_action_random_experiments",
        resume_ckpt=resume_ckpt,
        stage1_eval_samples=stage1_eval_samples,
    )

    log_stage("7/9 Initialize the Hugging Face Trainer")
    trainer = LocalActionTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )
    install_quiet_progress_callback(trainer)
    log_stage("8/9 Start model training")
    trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else None)


    log_stage("9/9 Save LoRA, implicit-state components and the merged model")
    peft_save_path = Path(cfg.SFT.output_dir) / f"{cfg.SFT.run_name}_local_action_peft_ckpts"
    if is_main_process:
        log.info(f"Saving local-action SFT-Random LoRA adapter to {peft_save_path}")
        save_local_action_model(model, str(peft_save_path))

    if dist.is_initialized():
        dist.barrier()


    merged_save_path = Path(cfg.SFT.output_dir) / f"{cfg.SFT.run_name}_local_action_merged_ckpts"
    if is_main_process and cfg.SFT.use_lora:
        log.info("Merging LoRA into base model for GRPO initialization...")
        unwrapped = trainer.accelerator.unwrap_model(model)
        merged_backbone = unwrapped.backbone.merge_and_unload()
        merged_backbone.save_pretrained(str(merged_save_path))
        log.info(f"Merged model saved to {merged_save_path}")


        unwrapped.save_local_components(str(merged_save_path))
        log.info("Local-action components and config saved alongside merged backbone")

    if dist.is_initialized():
        dist.barrier()

    if is_main_process:
        swanlab.finish()
        log.info("========== [DONE] Training and model saving completed ==========")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--processor_path", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eval_dataset", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--action_text_labels", required=True)
    parser.add_argument("--num_epochs", type=float, default=10)
    parser.add_argument("--stop_after_epochs", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1.5e-4)
    parser.add_argument("--eval_max_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--processor_use_fast", action="store_true")
    parser.add_argument("--optimal_bias_alpha", type=float, default=0.7)
    parser.add_argument("--action_output_mode", choices=("text_token", "linear"), default="text_token")
    parser.add_argument("--ablation_modules", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.optimal_bias_alpha <= 1:
        raise ValueError("optimal_bias_alpha must be in [0, 1]")
    sft = SimpleNamespace(
        model_path=args.model_path,
        qwen_processor_path=args.processor_path,
        qwen_visual_image_root=args.image_root,
        dataset_pth=args.dataset,
        eval_dataset_pth=args.eval_dataset,
        output_dir=args.output_dir,
        run_name=args.run_name,
        action_text_labels=args.action_text_labels,
        num_train_epochs=args.num_epochs,
        stop_after_epochs=args.stop_after_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_max_samples=args.eval_max_samples,
        seed=args.seed,
        qwen_processor_use_fast=args.processor_use_fast,
        backbone_type="qwen25vl",
        qwen_visual_input=True,
        qwen_visual_image_size=256,
        training_mode="local_action_random",
        sft_random_loss="optimal_biased_target_closer_not_closer",
        sft_random_optimal_bias_alpha=args.optimal_bias_alpha,
        ablation_modules=args.ablation_modules,
        state_flag_fusion_mode="add",
        state_token_mode="append",
        action_output_mode=args.action_output_mode,
        use_lora=True,
        freeze_backbone=True,
        dtype="bfloat16",
        tf32=True,
        gradient_accumulation_steps=1,
        position_num_frequencies=8,
        position_mlp_hidden_size=512,
        global_start_token_id=8193,
        global_end_token_id=8194,
        optim="adamw_torch_fused",
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
    )
    cfg = SimpleNamespace(SFT=sft)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    log_stage("Initialize the Python training process")
    seed_everything(sft.seed)
    configure_cuda_backend(sft.tf32)
    log.info("Training config: %s", sft)
    train_local_action_sft_random(cfg, dtype_map[sft.dtype])

if __name__ == "__main__":
    main()
