from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ssvr.reasoning import (
    is_maze_flip_meta,
    maze_flip_view_action_to_world,
    normalize_state,
    state_carrying,
    state_to_task_position,
    task_distance_to_goal,
    transition_step,
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_gib(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024 ** 3


def _measure(
    operation: Callable[[], Any], device: torch.device, warmup: int, repeats: int
) -> Dict[str, float]:
    for _ in range(warmup):
        operation()
    _synchronize(device)
    latencies = []
    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter()
        operation()
        _synchronize(device)
        latencies.append(1000 * (time.perf_counter() - start))
    ordered = sorted(latencies)
    return {
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p90_ms": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "std_ms": statistics.pstdev(latencies),
    }


def _profile_flops(operation: Callable[[], Any], device: torch.device) -> Dict[str, Any]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.profiler.profile(activities=activities, with_flops=True) as profile:
            operation()
            _synchronize(device)
        flops = sum(event.flops or 0 for event in profile.key_averages())
        return {"enabled": True, "profiler_supported_ops_tflops": flops / 1e12}
    except (RuntimeError, NotImplementedError) as exc:
        return {"enabled": False, "error": str(exc)}


@torch.no_grad()
def benchmark_gru_efficiency(
    model: torch.nn.Module,
    sample: Mapping[str, Any],
    qwen_inputs: Mapping[str, torch.Tensor],
    device: torch.device | str,
    *,
    warmup: int = 3,
    repeats: int = 20,
    compare_kv_cache: bool = True,
    profile_flops: bool = False,
) -> Dict[str, Any]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("Require warmup >= 0 and repeats > 0")
    device = torch.device(device)
    meta = sample["meta"]
    start_state = normalize_state(sample["input_state"], meta)
    expected_distance = task_distance_to_goal(meta, start_state)
    if expected_distance is None or expected_distance < 0:
        raise ValueError("Sample start state requires a valid distance to the goal")
    max_steps = int(expected_distance)
    height = int(meta.get("height", meta["level"]))
    width = int(meta.get("width", meta["level"]))
    global_input_ids = torch.tensor(sample["input_tokens"], dtype=torch.long, device=device).unsqueeze(0)
    map_size = torch.tensor([[height, width]], dtype=torch.float32, device=device)
    visual_inputs = {key: value.to(device) for key, value in qwen_inputs.items()}
    was_training = model.training
    model.eval()

    def step(state, latent_state, index, prefix_cache=None):
        position = torch.tensor([state_to_task_position(meta, state)], dtype=torch.float32, device=device)
        state_flags = torch.tensor([[float(state_carrying(meta, state))]], dtype=torch.float32, device=device)
        state_step = torch.tensor([index], dtype=torch.float32, device=device)
        model_map_size = map_size
        if is_maze_flip_meta(meta):
            position = torch.zeros_like(position)
            state_flags = torch.zeros_like(state_flags)
            state_step = torch.zeros_like(state_step)
            model_map_size = torch.ones_like(map_size)
        return model.predict_action(
            global_input_ids=global_input_ids, position=position, map_size=model_map_size,
            state_flags=state_flags, state_step=state_step, latent_state=latent_state,
            use_cache=prefix_cache is not None, qwen_prefix_cache=prefix_cache,
            **({} if prefix_cache is not None else visual_inputs),
        )

    def rollout(prefix_cache=None):
        state = start_state
        latent_state = model.initialize_latent_state(global_input_ids)
        actions = []
        for index in range(max_steps):
            action, _ = step(state, latent_state, index, prefix_cache)
            action_id = int(action.item())
            actions.append(action_id)
            latent_state = model.update_latent_state(latent_state, action)
            world_action = maze_flip_view_action_to_world(meta, state, action_id)
            result = transition_step(meta, state, world_action)
            if not result.valid or result.reason == "hole":
                break
            state = int(result.next_state)
            if result.terminal:
                break
        return actions

    try:
        initial_latent = model.initialize_latent_state(global_input_ids)

        def measure_mode(prefix_cache=None):
            single = lambda: step(start_state, initial_latent, 0, prefix_cache)
            closed_loop = lambda: rollout(prefix_cache)
            _reset_peak_memory(device)
            single_latency = _measure(single, device, warmup, repeats)
            rollout_latency = _measure(closed_loop, device, warmup, repeats)
            peak_memory = _peak_memory_gib(device)
            actions = closed_loop()
            mean_ms = rollout_latency["mean_ms"]
            return {
                "peak_inference_memory_gib": peak_memory,
                "single_step_forward_latency": single_latency,
                "closed_loop_rollout_latency": rollout_latency,
                "mean_ms_per_planned_step": mean_ms / max(1, len(actions)),
                "estimated_closed_loop_throughput_cases_per_sec": 1000 / mean_ms if mean_ms else 0.0,
                "flops": _profile_flops(single, device) if profile_flops else {"enabled": False},
            }, actions

        no_cache, actions = measure_mode()
        cached = None
        comparison = None
        prefix_build_ms = None
        if compare_kv_cache:
            _synchronize(device)
            start = time.perf_counter()
            prefix_cache = model.encode_qwen_visual_cache(**visual_inputs)
            _synchronize(device)
            prefix_build_ms = 1000 * (time.perf_counter() - start)
            cached, cached_actions = measure_mode(prefix_cache)
            cached["actions"] = cached_actions
            cached["prefix_cache_build_ms"] = prefix_build_ms
            step_ms = cached["single_step_forward_latency"]["mean_ms"]
            rollout_ms = cached["closed_loop_rollout_latency"]["mean_ms"]
            comparison = {
                "single_step_speedup": no_cache["single_step_forward_latency"]["mean_ms"] / step_ms if step_ms else 0.0,
                "closed_loop_speedup": no_cache["closed_loop_rollout_latency"]["mean_ms"] / rollout_ms if rollout_ms else 0.0,
            }
        return {
            "device": str(device), "warmup": warmup, "repeats": repeats,
            "expected_move": max_steps, "planned_steps": len(actions), "actions": actions,
            "no_cache": no_cache, "kv_cache": cached, "comparison": comparison,
            "prefix_cache_build_ms": prefix_build_ms,
            "timing_scope": "Processor work is excluded; cached forward/rollout latency excludes prefix construction.",
        }
    finally:
        model.train(was_training)
