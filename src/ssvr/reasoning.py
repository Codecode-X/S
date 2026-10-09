import json
import hashlib
import logging
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import jsonlines
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers.modeling_outputs import SequenceClassifierOutput

from ssvr.backbone import backbone_hidden_size

log = logging.getLogger(__name__)


ACTION_NAMES: Tuple[str, ...] = ("UP", "DOWN", "LEFT", "RIGHT")
MINIBEHAVIOUR_ACTION_NAMES: Tuple[str, ...] = ACTION_NAMES + ("PICK", "DROP")
ACTION_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(ACTION_NAMES)}
MINIBEHAVIOUR_ACTION_TO_ID: Dict[str, int] = {
    name: idx for idx, name in enumerate(MINIBEHAVIOUR_ACTION_NAMES)
}
ID_TO_ACTION: Dict[int, str] = {idx: name for name, idx in ACTION_TO_ID.items()}
GLOBAL_START_TOKEN_ID = 8193
GLOBAL_END_TOKEN_ID = 8194
ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    ACTION_TO_ID["UP"]: (-1, 0),
    ACTION_TO_ID["DOWN"]: (1, 0),
    ACTION_TO_ID["LEFT"]: (0, -1),
    ACTION_TO_ID["RIGHT"]: (0, 1),
}
ACTION_ID_TO_NAME_LOWER = {
    ACTION_TO_ID["UP"]: "up",
    ACTION_TO_ID["DOWN"]: "down",
    ACTION_TO_ID["LEFT"]: "left",
    ACTION_TO_ID["RIGHT"]: "right",
}
ACTION_NAME_LOWER_TO_ID = {name: action_id for action_id, name in ACTION_ID_TO_NAME_LOWER.items()}
DELTA_TO_ACTION_ID: Dict[Tuple[int, int], int] = {
    delta: action_id for action_id, delta in ACTION_DELTAS.items()
}


def is_minibehaviour_meta(meta: Mapping[str, Any]) -> bool:
    return "printer_pos" in meta and "table_pos" in meta


def is_maze_flip_meta(meta: Mapping[str, Any]) -> bool:
    return str(meta.get("task_variant", "")).lower() == "maze_flip"


def action_names_for_meta(meta: Mapping[str, Any]) -> Tuple[str, ...]:
    return MINIBEHAVIOUR_ACTION_NAMES if is_minibehaviour_meta(meta) else ACTION_NAMES


def qwen_visual_prompt_text(meta: Mapping[str, Any], current_state: Optional[int] = None) -> str:

    action_names = ", ".join(action_names_for_meta(meta))

    if is_minibehaviour_meta(meta):
        task_text = (
            "Navigate to the printer, pick up the object, then navigate to the target table and drop it. "
        )
    elif is_maze_flip_meta(meta):
        task_text = (
            "Navigate through the maze to the goal without crossing walls. "
            "LEFT/RIGHT moves flip the image top-to-bottom; "
            "UP/DOWN moves flip it left-to-right. "
            "Output the action for the current step. "
        )
    elif (
        isinstance(meta.get("layout"), list)
        and meta.get("layout")
        and isinstance(meta["layout"][0], list)
        and meta["layout"][0]
        and isinstance(meta["layout"][0][0], Mapping)
    ):
        task_text = (
            "Navigate through the maze to the goal without crossing walls. "
        )
    else:
        task_text = (
            "Navigate across FrozenLake to the goal while avoiding holes. "
        )

    return (
        f"{task_text}"
        f"Valid action labels, in action-head order, are: {action_names}. "
        "Select exactly one next action. Do not explain."
    )


def qwen_initial_visual_state(meta: Mapping[str, Any], current_state: Optional[int] = None) -> Optional[int]:

    if "start_pos" not in meta:
        return current_state
    start_pos = meta["start_pos"]
    if is_minibehaviour_meta(meta):
        return task_state_from_position(meta, start_pos, False)
    if isinstance(start_pos, (list, tuple)):
        width = int(meta.get("width", meta["level"]))
        return position_to_state(int(start_pos[0]), int(start_pos[1]), width)
    return int(start_pos)


def visual_token_hash(tokens: Sequence[int]) -> str:

    digest = hashlib.sha1()
    digest.update(",".join(str(int(token)) for token in tokens).encode("utf-8"))
    return digest.hexdigest()


def qwen_visual_image_path(image_root: str | Path, tokens: Sequence[int]) -> Path:
    return Path(image_root) / f"{visual_token_hash(tokens)}.png"


def qwen_visual_image_from_tokens(
    image_root: str | Path,
    tokens: Sequence[int],
    scale: float = 1.0,
):

    image_path = qwen_visual_image_path(image_root, tokens)
    if not image_path.exists():
        raise FileNotFoundError(
            f"Missing Qwen visual image for token hash {visual_token_hash(tokens)}: "
            f"{image_path}. Build it with scripts/common/prepare_images.sh."
        )
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for Qwen visual-input image loading") from exc
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError(f"qwen visual image scale must be positive, got {scale}")
    if not math.isclose(scale, 1.0):
        width, height = rgb.size
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        rgb = rgb.resize(new_size, resampling)
    return rgb


def build_qwen_visual_start_tokens_index(
    raw_samples: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[Any, int], Tuple[int, ...]]:

    _, start_tokens_index = SFTOptimLocalActionDataset._build_indices(
        raw_samples,
        strict_token_state=False,
    )
    return dict(start_tokens_index)


def qwen_visual_tokens_for_sample(
    sample: Mapping[str, Any],
    start_tokens_index: Optional[Mapping[Tuple[Any, int], Tuple[int, ...]]] = None,
) -> Tuple[int, ...]:

    if "global_input_ids" in sample:
        value = sample["global_input_ids"]
        if isinstance(value, torch.Tensor):
            return tuple(int(token) for token in value.detach().cpu().tolist())
        return tuple(int(token) for token in value)
    if "global_tokens" in sample:
        return tuple(int(token) for token in sample["global_tokens"])

    meta = sample["meta"]
    current_state = normalize_state(sample.get("input_state", 0), meta)
    start_state = qwen_initial_visual_state(meta, current_state)
    if start_state is not None and start_tokens_index is not None:
        tokens = start_tokens_index.get((sample_group_key(meta), int(start_state)))
        if tokens is not None:
            return tuple(int(token) for token in tokens)

    if start_state is not None and current_state == int(start_state):
        return tuple(int(token) for token in sample["input_tokens"])
    raise ValueError(
        "Cannot recover initial visual tokens for Qwen image context. "
        "Pass a start_tokens_index built from the full JSONL dataset."
    )


def normalize_state(state: Any, meta: Optional[Mapping[str, Any]] = None) -> int:
    if isinstance(state, (list, tuple)):
        if not state:
            raise ValueError("input_state cannot be empty")
        if (
            isinstance(state[0], (list, tuple))
            and len(state[0]) == 2
            and isinstance(state[0][0], (list, tuple))
        ):
            return normalize_state(state[0], meta)
        if (
            len(state) == 2
            and isinstance(state[0], (list, tuple))
            and len(state[0]) == 2
        ):
            if meta is None:
                raise ValueError("MiniBehaviour state normalization requires metadata")
            width = int(meta.get("width", meta["level"]))
            cell = position_to_state(int(state[0][0]), int(state[0][1]), width)
            return cell * 2 + int(bool(state[1]))
        return int(state[0])
    return int(state)


def state_to_task_position(
    meta: Mapping[str, Any], state: int
) -> Tuple[int, int]:
    width = int(meta.get("width", meta["level"]))
    cell_state = int(state) // 2 if is_minibehaviour_meta(meta) else int(state)
    return state_to_position(cell_state, width)


def state_carrying(meta: Mapping[str, Any], state: int) -> bool:
    return bool(int(state) % 2) if is_minibehaviour_meta(meta) else False


def task_state_from_position(
    meta: Mapping[str, Any], position: Sequence[int], carrying: bool = False
) -> int:
    width = int(meta.get("width", meta["level"]))
    cell_state = position_to_state(int(position[0]), int(position[1]), width)
    return cell_state * 2 + int(bool(carrying)) if is_minibehaviour_meta(meta) else cell_state


def state_to_position(state: int, width: int) -> Tuple[int, int]:
    if width <= 0:
        raise ValueError("width must be positive")
    return int(state) // width, int(state) % width


def position_to_state(row: int, col: int, width: int) -> int:
    if width <= 0:
        raise ValueError("width must be positive")
    return int(row) * width + int(col)


def infer_action_id_from_states(current_state: int, next_state: int, width: int) -> int:
    current_pos = state_to_position(current_state, width)
    next_pos = state_to_position(next_state, width)
    delta = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
    if delta not in DELTA_TO_ACTION_ID:
        raise ValueError(
            f"Cannot infer a single-step action from state {current_state} to {next_state}"
        )
    return DELTA_TO_ACTION_ID[delta]


def infer_task_action_id_from_states(
    meta: Mapping[str, Any], current_state: int, next_state: int
) -> int:
    current_pos = state_to_task_position(meta, current_state)
    next_pos = state_to_task_position(meta, next_state)
    current_carrying = state_carrying(meta, current_state)
    next_carrying = state_carrying(meta, next_state)
    if current_pos == next_pos and not current_carrying and next_carrying:
        return MINIBEHAVIOUR_ACTION_TO_ID["PICK"]
    if current_pos == next_pos and current_carrying and not next_carrying:
        return MINIBEHAVIOUR_ACTION_TO_ID["DROP"]
    width = int(meta.get("width", meta["level"]))
    current_cell = position_to_state(*current_pos, width)
    next_cell = position_to_state(*next_pos, width)
    return infer_action_id_from_states(current_cell, next_cell, width)


def maze_flip_view_for_state(meta: Mapping[str, Any], state: int) -> Tuple[bool, bool]:

    if not is_maze_flip_meta(meta):
        return False, False
    width = int(meta.get("width", meta["level"]))
    start_row, start_col = state_to_position(int(meta["start_pos"]), width)
    row, col = state_to_position(int(state), width)
    return bool((col - start_col) % 2), bool((row - start_row) % 2)


def maze_flip_world_action_to_view(
    meta: Mapping[str, Any], state: int, world_action_id: int
) -> int:
    action_id = int(world_action_id)
    if not is_maze_flip_meta(meta):
        return action_id
    flip_vertical, flip_horizontal = maze_flip_view_for_state(meta, state)
    if flip_vertical and action_id in (ACTION_TO_ID["UP"], ACTION_TO_ID["DOWN"]):
        action_id = ACTION_TO_ID["DOWN"] if action_id == ACTION_TO_ID["UP"] else ACTION_TO_ID["UP"]
    if flip_horizontal and action_id in (ACTION_TO_ID["LEFT"], ACTION_TO_ID["RIGHT"]):
        action_id = ACTION_TO_ID["RIGHT"] if action_id == ACTION_TO_ID["LEFT"] else ACTION_TO_ID["LEFT"]
    return action_id


def maze_flip_view_action_to_world(
    meta: Mapping[str, Any], state: int, view_action_id: int
) -> int:

    return maze_flip_world_action_to_view(meta, state, view_action_id)


def maze_flip_world_mask_to_view(
    meta: Mapping[str, Any], state: int, world_mask: Sequence[float]
) -> Tuple[float, ...]:
    if not is_maze_flip_meta(meta):
        return tuple(float(value) for value in world_mask)
    view_mask = [0.0 for _ in world_mask]
    for world_action_id, value in enumerate(world_mask):
        view_action_id = maze_flip_world_action_to_view(meta, state, world_action_id)
        view_mask[view_action_id] = float(value)
    return tuple(view_mask)


def _layout_key(layout: Any) -> str:
    return json.dumps(layout, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sample_group_key(meta: Mapping[str, Any]) -> Tuple[Any, ...]:
    if is_minibehaviour_meta(meta):
        internal_group = meta.get("_local_action_group_id")
        if internal_group is not None:
            return ("minibehaviour", int(internal_group))
        static_meta = {
            key: meta.get(key)
            for key in (
                "level",
                "start_pos",
                "start_dir",
                "printer_pos",
                "table_pos",
                "printer_neighbors",
                "table_neighbors",
                "distance_map_to_printer",
                "distance_map_to_table",
            )
        }
        return ("minibehaviour", _layout_key(static_meta))
    return (
        int(meta["level"]),
        int(meta.get("start_pos", -1)),
        int(meta.get("target_pos", -1)),
        _layout_key(meta.get("layout")),
    )


def task_start_state(meta: Mapping[str, Any], fallback_state: int) -> int:
    if is_minibehaviour_meta(meta):
        return task_state_from_position(meta, meta["start_pos"], False)
    return int(meta.get("start_pos", fallback_state))


def ordered_step_index(
    step_counters: Dict[Tuple[Any, ...], int],
    meta: Mapping[str, Any],
    current_state: int,
) -> int:

    group = sample_group_key(meta)
    start_state = task_start_state(meta, current_state)
    step_index = (
        0
        if group not in step_counters or int(current_state) == int(start_state)
        else int(step_counters[group])
    )
    step_counters[group] = int(step_index) + 1
    return int(step_index)


def iter_neighbor_states(state: int, height: int, width: int) -> Iterable[Tuple[int, int]]:
    row, col = state_to_position(state, width)
    for action_id, (dr, dc) in ACTION_DELTAS.items():
        next_row, next_col = row + dr, col + dc
        if 0 <= next_row < height and 0 <= next_col < width:
            yield action_id, position_to_state(next_row, next_col, width)


def is_maze_layout(layout: Any) -> bool:
    return (
        isinstance(layout, list)
        and bool(layout)
        and isinstance(layout[0], list)
        and bool(layout[0])
        and isinstance(layout[0][0], Mapping)
    )


def is_frozenlake_layout(layout: Any) -> bool:
    return (
        isinstance(layout, list)
        and bool(layout)
        and isinstance(layout[0], list)
        and bool(layout[0])
        and isinstance(layout[0][0], str)
    )


@dataclass(frozen=True)
class TransitionResult:
    next_state: int
    valid: bool
    terminal: bool
    reason: str


def _coord_set(values: Any) -> set[Tuple[int, int]]:
    return {tuple(int(v) for v in coord) for coord in values or []}


def minibehaviour_walkable_positions(
    meta: Mapping[str, Any], carrying: bool = False
) -> set[Tuple[int, int]]:
    cache_key = (
        "_local_action_walkable_positions_carrying"
        if carrying
        else "_local_action_walkable_positions_not_carrying"
    )
    cached = meta.get(cache_key)
    if cached is not None:
        return cached
    positions: set[Tuple[int, int]] = set()
    for maps_name in ("distance_map_to_printer", "distance_map_to_table"):
        for distance_map in meta.get(maps_name, {}).values():
            for key in distance_map:
                row, col = key.strip("()").split(",")
                positions.add((int(row), int(col)))
    positions.add(tuple(int(v) for v in meta["start_pos"]))
    occupied_positions = {
        tuple(int(v) for v in table_position)
        for table_position in meta.get("table_pos", [])
    }


    if not carrying:
        occupied_positions.add(tuple(int(v) for v in meta["printer_pos"]))
    positions.difference_update(occupied_positions)
    if isinstance(meta, dict):
        meta[cache_key] = positions
    return positions


def is_action_legal(meta: Mapping[str, Any], state: int, action_id: int) -> bool:
    level = int(meta["level"])
    height = int(meta.get("height", level))
    width = int(meta.get("width", level))
    if is_minibehaviour_meta(meta):
        row, col = state_to_task_position(meta, state)
        carrying = state_carrying(meta, state)
        if action_id == MINIBEHAVIOUR_ACTION_TO_ID["PICK"]:
            return not carrying and (row, col) in _coord_set(meta.get("printer_neighbors"))
        if action_id == MINIBEHAVIOUR_ACTION_TO_ID["DROP"]:
            return carrying and (row, col) in _coord_set(meta.get("table_neighbors"))
        if action_id not in ACTION_DELTAS:
            return False
        dr, dc = ACTION_DELTAS[action_id]
        return (row + dr, col + dc) in minibehaviour_walkable_positions(
            meta, carrying=carrying
        )

    if action_id not in ACTION_DELTAS:
        return False
    row, col = state_to_position(state, width)
    dr, dc = ACTION_DELTAS[action_id]
    next_row, next_col = row + dr, col + dc
    if not (0 <= next_row < height and 0 <= next_col < width):
        return False

    layout = meta.get("layout")
    if is_maze_layout(layout):
        direction = ACTION_ID_TO_NAME_LOWER[action_id]
        wall_key = {
            "up": "north",
            "down": "south",
            "left": "west",
            "right": "east",
        }[direction]
        return not bool(layout[row][col][wall_key])

    return True


def is_terminal_state(meta: Mapping[str, Any], state: int) -> Tuple[bool, str]:
    if is_minibehaviour_meta(meta):
        return False, "none"
    level = int(meta["level"])
    width = int(meta.get("width", level))
    target_pos = int(meta.get("target_pos", -1))
    if state == target_pos:
        return True, "target"

    layout = meta.get("layout")
    if is_frozenlake_layout(layout):
        row, col = state_to_position(state, width)
        if layout[row][col] == "H":
            return True, "hole"

    return False, "none"


def transition_step(meta: Mapping[str, Any], state: int, action_id: int) -> TransitionResult:
    level = int(meta["level"])
    width = int(meta.get("width", level))
    if not is_action_legal(meta, state, action_id):
        return TransitionResult(next_state=state, valid=False, terminal=False, reason="invalid")
    if is_minibehaviour_meta(meta):
        row, col = state_to_task_position(meta, state)
        carrying = state_carrying(meta, state)
        if action_id == MINIBEHAVIOUR_ACTION_TO_ID["PICK"]:
            return TransitionResult(
                next_state=task_state_from_position(meta, (row, col), True),
                valid=True,
                terminal=False,
                reason="pick",
            )
        if action_id == MINIBEHAVIOUR_ACTION_TO_ID["DROP"]:
            return TransitionResult(
                next_state=task_state_from_position(meta, (row, col), False),
                valid=True,
                terminal=True,
                reason="target",
            )
        dr, dc = ACTION_DELTAS[action_id]
        return TransitionResult(
            next_state=task_state_from_position(meta, (row + dr, col + dc), carrying),
            valid=True,
            terminal=False,
            reason="none",
        )
    row, col = state_to_position(state, width)
    dr, dc = ACTION_DELTAS[action_id]
    next_state = position_to_state(row + dr, col + dc, width)
    terminal, reason = is_terminal_state(meta, next_state)
    return TransitionResult(next_state=next_state, valid=True, terminal=terminal, reason=reason)


def transition_state(meta: Mapping[str, Any], state: int, action_id: int) -> Tuple[int, bool]:
    result = transition_step(meta, state, action_id)
    return result.next_state, result.valid


def canonical_action_history(meta: Mapping[str, Any], target_state: int) -> Tuple[int, ...]:

    start_state = task_start_state(meta, int(target_state))
    target_state = int(target_state)
    if start_state == target_state:
        return tuple()
    queue = deque([(start_state, tuple())])
    visited = {start_state}
    while queue:
        state, history = queue.popleft()
        for action_id in range(len(action_names_for_meta(meta))):
            result = transition_step(meta, state, action_id)
            if not result.valid or result.reason == "hole":
                continue
            next_state = int(result.next_state)
            next_history = history + (action_id,)
            if next_state == target_state:
                return next_history
            if result.terminal or next_state in visited:
                continue
            visited.add(next_state)
            queue.append((next_state, next_history))
    raise ValueError(f"State {target_state} is unreachable from start state {start_state}")


def model_action_history(meta: Mapping[str, Any], target_state: int) -> Tuple[int, ...]:

    world_history = canonical_action_history(meta, target_state)
    if not is_maze_flip_meta(meta):
        return world_history
    state = task_start_state(meta, int(target_state))
    view_history: List[int] = []
    for world_action_id in world_history:
        view_history.append(maze_flip_world_action_to_view(meta, state, world_action_id))
        result = transition_step(meta, state, world_action_id)
        if not result.valid:
            raise ValueError("Canonical MazeFlip history contains an invalid world transition")
        state = int(result.next_state)
    if state != int(target_state):
        raise ValueError("Canonical MazeFlip history does not reach target_state")
    return tuple(view_history)


def valid_action_mask(meta: Mapping[str, Any], state: int) -> Tuple[float, ...]:
    mask = tuple(
        1.0 if is_action_legal(meta, state, action_id) else 0.0
        for action_id in range(len(action_names_for_meta(meta)))
    )
    if sum(mask) <= 0:
        raise ValueError(f"State {state} has no legal actions")
    return mask


def target_closer_not_closer_action_mask(meta: Mapping[str, Any], state: int) -> Tuple[float, ...]:
    if is_minibehaviour_meta(meta):
        current_distance = task_distance_to_goal(meta, state)
        mask = []
        for action_id in range(len(action_names_for_meta(meta))):
            result = transition_step(meta, state, action_id)
            if not result.valid:
                mask.append(0.0)
                continue
            if result.reason in {"target", "pick"}:
                mask.append(1.0)
                continue
            next_distance = task_distance_to_goal(meta, result.next_state)
            if (
                current_distance is None
                or next_distance is None
                or next_distance == current_distance
            ):
                mask.append(0.0)
                continue
            mask.append(1.0)
        return tuple(mask)
    distance_map = meta.get("distance_map", {})
    current_distance = distance_map.get(str(state))
    mask: List[float] = []
    for action_id in range(len(ACTION_NAMES)):
        result = transition_step(meta, state, action_id)
        if not result.valid or result.reason == "hole":
            mask.append(0.0)
            continue
        if result.reason == "target":
            mask.append(1.0)
            continue
        next_distance = distance_map.get(str(result.next_state))
        if (
            current_distance is None
            or next_distance is None
            or current_distance == -1
            or next_distance == -1
            or next_distance == current_distance
        ):
            mask.append(0.0)
            continue
        mask.append(1.0)
    return tuple(mask)


def optimal_biased_target_closer_not_closer_action_mask(
    meta: Mapping[str, Any],
    state: int,
    alpha: float = 0.7,
) -> Tuple[float, ...]:

    alpha = float(alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0, 1]")

    base_mask = target_closer_not_closer_action_mask(meta, state)
    base_count = float(sum(base_mask))
    if base_count <= 0.0:
        return base_mask

    current_distance = task_distance_to_goal(meta, state)
    optimal_mask: List[float] = []
    for action_id in range(len(action_names_for_meta(meta))):
        result = transition_step(meta, state, action_id)
        if not result.valid or result.reason == "hole":
            optimal_mask.append(0.0)
            continue
        if result.reason in {"target", "pick"}:
            optimal_mask.append(1.0)
            continue
        next_distance = task_distance_to_goal(meta, result.next_state)
        if (
            current_distance is not None
            and next_distance is not None
            and float(current_distance) != -1.0
            and float(next_distance) == float(current_distance) - 1.0
        ):
            optimal_mask.append(1.0)
        else:
            optimal_mask.append(0.0)

    optimal_count = float(sum(optimal_mask))
    if optimal_count <= 0.0:
        return base_mask

    return tuple(
        (1.0 - alpha) * (float(base_enabled) / base_count)
        + alpha * (float(optimal_enabled) / optimal_count)
        for base_enabled, optimal_enabled in zip(base_mask, optimal_mask)
    )


def sft_random_action_mask(
    meta: Mapping[str, Any],
    state: int,
    optimal_bias_alpha: float = 0.7,
) -> Tuple[float, ...]:
    world_mask = optimal_biased_target_closer_not_closer_action_mask(
        meta,
        state,
        alpha=optimal_bias_alpha,
    )
    return maze_flip_world_mask_to_view(meta, state, world_mask)


def task_distance_to_goal(meta: Mapping[str, Any], state: int) -> Optional[float]:
    if is_minibehaviour_meta(meta):
        coord_key = str(tuple(state_to_task_position(meta, state)))
        if not state_carrying(meta, state):
            best_total = float("inf")
            for printer_neighbor in meta.get("printer_neighbors", []):
                printer_key = str(tuple(printer_neighbor))
                current_distance = meta["distance_map_to_printer"].get(
                    printer_key, {}
                ).get(coord_key)
                if current_distance is None:
                    continue
                for table_neighbor in meta.get("table_neighbors", []):
                    table_key = str(tuple(table_neighbor))
                    onward = meta["distance_map_to_table"].get(
                        table_key, {}
                    ).get(printer_key)
                    if onward is not None:
                        best_total = min(
                            best_total,
                            float(current_distance + onward + 2),
                        )
            return None if best_total == float("inf") else best_total
        best_table = min(
            (
                float(
                    meta["distance_map_to_table"][str(tuple(target))][coord_key]
                    + 1
                )
                for target in meta.get("table_neighbors", [])
                if coord_key
                in meta["distance_map_to_table"].get(str(tuple(target)), {})
            ),
            default=float("inf"),
        )
        return None if best_table == float("inf") else best_table
    distance = meta.get("distance_map", {}).get(str(int(state)))
    return None if distance is None else float(distance)


def infer_best_next_state_from_distance_map(
    meta: Mapping[str, Any], current_state: int
) -> Tuple[int, int]:
    if is_minibehaviour_meta(meta):
        raise ValueError("MiniBehaviour optimal labels must be recovered from trajectory tokens")
    level = int(meta["level"])
    height = int(meta.get("height", level))
    width = int(meta.get("width", level))
    target_pos = int(meta["target_pos"])
    distance_map = {int(k): int(v) for k, v in meta["distance_map"].items()}

    if current_state not in distance_map:
        raise ValueError(f"State {current_state} is absent from distance_map")

    current_distance = distance_map[current_state]
    candidates: List[Tuple[int, int]] = []
    for action_id, next_state in iter_neighbor_states(current_state, height, width):
        if not is_action_legal(meta, current_state, action_id):
            continue
        if next_state == target_pos:
            next_distance = 0
        else:
            next_distance = distance_map.get(next_state)
        if next_distance == current_distance - 1:
            candidates.append((action_id, next_state))

    if not candidates:
        raise ValueError(f"No distance-improving neighbor found for state {current_state}")

    action_id, next_state = candidates[0]
    return next_state, action_id


def infer_terminal_action(meta: Mapping[str, Any], current_state: int) -> Optional[Tuple[int, int]]:
    width = int(meta.get("width", int(meta["level"])))
    candidates: List[Tuple[int, int]] = []
    for action_id in range(len(action_names_for_meta(meta))):
        result = transition_step(meta, current_state, action_id)
        if result.valid and result.terminal:
            candidates.append((action_id, result.next_state))
    if len(candidates) == 1:
        return candidates[0]


    return None


def recover_minibehaviour_action_from_output_state(
    obj: Mapping[str, Any], current_state: int
) -> Optional[Tuple[int, int]]:
    meta = obj["meta"]
    if not is_minibehaviour_meta(meta) or "output_state" not in obj:
        return None
    next_state = normalize_state(obj["output_state"], meta)
    action_id = infer_task_action_id_from_states(meta, current_state, next_state)
    result = transition_step(meta, current_state, action_id)
    if not result.valid or result.next_state != next_state:
        raise ValueError(
            "MiniBehaviour output_state is not reachable by its inferred action: "
            f"current_state={current_state}, output_state={next_state}, "
            f"action_id={action_id}, transition={result}"
        )
    return action_id, next_state


@dataclass(frozen=True)
class LocalActionSample:
    global_tokens: Tuple[int, ...]
    current_position: Tuple[int, int]
    map_size: Tuple[int, int]
    target_action_id: int
    current_state: int
    next_state: int
    state_step: int
    state_flags: Tuple[float, ...]
    action_source: str
    meta: Mapping[str, Any]
    state_action_history: Tuple[int, ...]


class SFTOptimLocalActionDataset(Dataset):

    def __init__(
        self,
        filepath: str,
        require_exact_next_state: bool = False,
        require_action_labels: bool = True,
    ):
        self.filepath = str(filepath)
        self.require_exact_next_state = require_exact_next_state
        self.require_action_labels = bool(require_action_labels)
        raw_samples = self._read_raw_samples(filepath)
        token_state_index, start_tokens_index = self._build_indices(raw_samples)
        if self.require_action_labels:
            self.samples = self._build_samples(
                raw_samples=raw_samples,
                token_state_index=token_state_index,
                start_tokens_index=start_tokens_index,
            )
        else:
            self.samples = self._build_prompt_samples(
                raw_samples=raw_samples,
                start_tokens_index=start_tokens_index,
            )
        if not self.samples:
            raise ValueError(f"No usable SFT-Optimal samples were built from {filepath}")
        self.action_names = action_names_for_meta(self.samples[0].meta)
        for sample in self.samples:
            sample_action_names = action_names_for_meta(sample.meta)
            if sample_action_names != self.action_names:
                raise ValueError(
                    "SFT-Optimal dataset mixes incompatible action spaces: "
                    f"{self.action_names} and {sample_action_names}"
                )

    @staticmethod
    def _read_raw_samples(filepath: str) -> List[Dict[str, Any]]:
        with open(filepath, "r", encoding="utf-8") as raw_file:
            first_line = raw_file.readline()
        if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
            raise ValueError(
                f"Dataset is a Git LFS pointer, not materialized JSONL: {filepath}. "
                f"Run: git -C {Path(filepath).parent} lfs pull"
            )
        samples: List[Dict[str, Any]] = []
        previous_mini_static: Optional[Tuple[Any, ...]] = None
        mini_group_id = -1
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                meta = obj.get("meta", {})
                if is_minibehaviour_meta(meta):
                    mini_static = tuple(
                        meta.get(key)
                        for key in (
                            "level",
                            "start_pos",
                            "start_dir",
                            "printer_pos",
                            "table_pos",
                            "printer_neighbors",
                            "table_neighbors",
                            "distance_map_to_printer",
                            "distance_map_to_table",
                        )
                    )
                    if previous_mini_static is None or mini_static != previous_mini_static:
                        mini_group_id += 1
                    meta["_local_action_group_id"] = mini_group_id
                    previous_mini_static = mini_static
                samples.append(obj)
        if not samples:
            raise ValueError(f"Dataset is empty: {filepath}")
        return samples

    @staticmethod
    def _build_indices(
        raw_samples: Sequence[Mapping[str, Any]],
        strict_token_state: bool = True,
    ) -> Tuple[Dict[Tuple[Any, Tuple[int, ...]], int], Dict[Tuple[Any, int], Tuple[int, ...]]]:
        token_state_index: Dict[Tuple[Any, Tuple[int, ...]], int] = {}
        state_tokens_index: Dict[Tuple[Any, int], Tuple[int, ...]] = {}
        ambiguous_token_keys: set[Tuple[Any, Tuple[int, ...]]] = set()

        for obj in raw_samples:
            meta = obj["meta"]
            group = sample_group_key(meta)
            current_state = normalize_state(obj["input_state"], meta)
            input_tokens = tuple(int(t) for t in obj["input_tokens"])

            token_key = (group, input_tokens)
            if token_key in ambiguous_token_keys:
                state_tokens_index.setdefault((group, current_state), input_tokens)
                continue
            existing_state = token_state_index.get(token_key)
            if existing_state is not None and existing_state != current_state:
                if strict_token_state:
                    raise ValueError(
                        "The same visual token sequence maps to multiple states "
                        f"inside group {group}: {existing_state} and {current_state}"
                    )
                ambiguous_token_keys.add(token_key)
                token_state_index.pop(token_key, None)
            else:
                token_state_index[token_key] = current_state


            state_tokens_index.setdefault((group, current_state), input_tokens)

        start_tokens_index: Dict[Tuple[Any, int], Tuple[int, ...]] = {}
        for obj in raw_samples:
            meta = obj["meta"]
            group = sample_group_key(meta)
            start_state = (
                task_state_from_position(meta, meta["start_pos"], False)
                if is_minibehaviour_meta(meta)
                else int(meta.get("start_pos", normalize_state(obj["input_state"], meta)))
            )
            start_tokens = state_tokens_index.get((group, start_state))
            if start_tokens is not None:
                start_tokens_index[(group, start_state)] = start_tokens

        return token_state_index, start_tokens_index

    def _build_samples(
        self,
        raw_samples: Sequence[Mapping[str, Any]],
        token_state_index: Mapping[Tuple[Any, Tuple[int, ...]], int],
        start_tokens_index: Mapping[Tuple[Any, int], Tuple[int, ...]],
    ) -> List[LocalActionSample]:
        samples: List[LocalActionSample] = []
        step_counters: Dict[Tuple[Any, ...], int] = {}

        for obj in raw_samples:
            meta = obj["meta"]
            group = sample_group_key(meta)
            level = int(meta["level"])
            height = int(meta.get("height", level))
            width = int(meta.get("width", level))
            current_state = normalize_state(obj["input_state"], meta)
            state_step = ordered_step_index(step_counters, meta, current_state)
            output_tokens = tuple(int(t) for t in obj["output_tokens"])

            recovered = recover_minibehaviour_action_from_output_state(obj, current_state)
            if recovered is not None:
                action_id, next_state = recovered
                action_source = "output_state"
            else:
                next_state = token_state_index.get((group, output_tokens))
                if next_state is not None:
                    action_id = infer_task_action_id_from_states(meta, current_state, next_state)
                    action_source = "token_match"
                else:
                    terminal_result = (
                        infer_terminal_action(meta, current_state)
                        if is_minibehaviour_meta(meta)
                        else None
                    )
                    if terminal_result is not None:
                        action_id, next_state = terminal_result
                        action_source = "terminal_recovery"
                    elif self.require_exact_next_state or is_minibehaviour_meta(meta):
                        raise ValueError(
                            "Could not recover next_state by matching output_tokens "
                            f"for current_state={current_state}"
                        )
                    else:
                        next_state, action_id = infer_best_next_state_from_distance_map(
                            meta, current_state
                        )
                        action_source = "distance_map"

            start_state = (
                task_state_from_position(meta, meta["start_pos"], False)
                if is_minibehaviour_meta(meta)
                else int(meta.get("start_pos", current_state))
            )
            global_tokens = start_tokens_index.get((group, start_state))
            if global_tokens is None:
                raise ValueError(
                    f"Could not find global start tokens for start_state={start_state} "
                    f"inside group {group}"
                )

            samples.append(
                LocalActionSample(
                    global_tokens=global_tokens,
                    current_position=state_to_task_position(meta, current_state),
                    map_size=(height, width),
                    target_action_id=action_id,
                    current_state=current_state,
                    next_state=next_state,
                    state_step=state_step,
                    state_flags=(
                        (float(state_carrying(meta, current_state)),)
                        if is_minibehaviour_meta(meta)
                        else tuple()
                    ),
                    action_source=action_source,
                    meta=meta,
                    state_action_history=model_action_history(meta, current_state),
                )
            )

        return samples

    def _build_prompt_samples(
        self,
        raw_samples: Sequence[Mapping[str, Any]],
        start_tokens_index: Mapping[Tuple[Any, int], Tuple[int, ...]],
    ) -> List[LocalActionSample]:
        samples: List[LocalActionSample] = []
        step_counters: Dict[Tuple[Any, ...], int] = {}
        for obj in raw_samples:
            meta = obj["meta"]
            group = sample_group_key(meta)
            level = int(meta["level"])
            height = int(meta.get("height", level))
            width = int(meta.get("width", level))
            current_state = normalize_state(obj["input_state"], meta)
            state_step = ordered_step_index(step_counters, meta, current_state)
            start_state = (
                task_state_from_position(meta, meta["start_pos"], False)
                if is_minibehaviour_meta(meta)
                else int(meta.get("start_pos", current_state))
            )
            global_tokens = start_tokens_index.get(
                (group, start_state),
                tuple(int(token) for token in obj["input_tokens"]),
            )
            samples.append(
                LocalActionSample(
                    global_tokens=global_tokens,
                    current_position=state_to_task_position(meta, current_state),
                    map_size=(height, width),
                    target_action_id=0,
                    current_state=current_state,
                    next_state=current_state,
                    state_step=state_step,
                    state_flags=(
                        (float(state_carrying(meta, current_state)),)
                        if is_minibehaviour_meta(meta)
                        else tuple()
                    ),
                    action_source="prompt_only",
                    meta=meta,
                    state_action_history=model_action_history(meta, current_state),
                )
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]


        if is_maze_flip_meta(sample.meta):
            model_position = torch.zeros(2, dtype=torch.float32)
            model_map_size = torch.ones(2, dtype=torch.float32)
            model_state_step = torch.tensor(0.0, dtype=torch.float32)
        else:
            model_position = torch.tensor(sample.current_position, dtype=torch.float32)
            model_map_size = torch.tensor(sample.map_size, dtype=torch.float32)
            model_state_step = torch.tensor(sample.state_step, dtype=torch.float32)
        item = {
            "global_input_ids": torch.tensor(sample.global_tokens, dtype=torch.long),
            "position": model_position,
            "map_size": model_map_size,
            "labels": torch.tensor(sample.target_action_id, dtype=torch.long),
            "state_step": model_state_step,
            "current_state": sample.current_state,
            "next_state": sample.next_state,
            "action_source": sample.action_source,
            "state_action_history": torch.tensor(
                sample.state_action_history, dtype=torch.long
            ),
            "meta": sample.meta,
        }
        if sample.state_flags:
            item["state_flags"] = torch.tensor(sample.state_flags, dtype=torch.float32)
        return item


def local_action_collate_fn(
    batch: Sequence[Mapping[str, Any]],
    include_valid_action_mask: bool = False,
) -> Dict[str, Any]:
    histories = [item["state_action_history"] for item in batch]
    max_history = max((int(history.numel()) for history in histories), default=0)
    padded_histories = torch.zeros((len(batch), max_history), dtype=torch.long)
    history_mask = torch.zeros((len(batch), max_history), dtype=torch.bool)
    for index, history in enumerate(histories):
        length = int(history.numel())
        if length:
            padded_histories[index, :length] = history
            history_mask[index, :length] = True
    collated = {
        "global_input_ids": torch.stack([item["global_input_ids"] for item in batch], dim=0),
        "position": torch.stack([item["position"] for item in batch], dim=0),
        "map_size": torch.stack([item["map_size"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "state_step": torch.stack([item["state_step"] for item in batch], dim=0),
        "state_action_history": padded_histories,
        "state_action_history_mask": history_mask,
    }
    if all("state_flags" in item for item in batch):
        collated["state_flags"] = torch.stack([item["state_flags"] for item in batch], dim=0)
    if include_valid_action_mask:
        collated["valid_action_mask"] = torch.stack(
            [item["valid_action_mask"] for item in batch],
            dim=0,
        )
    return collated


def local_action_valid_mask_collate_fn(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return local_action_collate_fn(
        batch,
        include_valid_action_mask=True,
    )


class QwenVisualLocalActionCollator:

    def __init__(
        self,
        processor_path: str,
        include_valid_action_mask: bool = False,
        image_size: int = 256,
        processor_use_fast: bool = False,
        image_root: Optional[str] = None,
        prompt_text: Optional[str] = None,
    ):
        self.processor_path = str(processor_path)
        self.include_valid_action_mask = bool(include_valid_action_mask)
        self.image_size = int(image_size)
        self.processor_use_fast = bool(processor_use_fast)
        self.image_root = None if image_root in (None, "") else str(image_root)
        if self.image_root is None:
            raise ValueError(
                "Qwen native visual input requires qwen_visual_image_root with "
                "VQ-decoded initial-map images."
            )
        self.prompt_text = None if prompt_text is None else str(prompt_text)
        self._processor = None

    @property
    def processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.processor_path,
                trust_remote_code=True,
                use_fast=self.processor_use_fast,
            )
        return self._processor

    def _prompt(self, meta: Mapping[str, Any], current_state: int) -> str:
        prompt_text = self.prompt_text or qwen_visual_prompt_text(meta, current_state)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        apply_template = getattr(self.processor, "apply_chat_template", None)
        if callable(apply_template):
            return apply_template(messages, tokenize=False, add_generation_prompt=True)
        return f"<image>\n{prompt_text}"

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        collated = local_action_collate_fn(
            batch,
            include_valid_action_mask=self.include_valid_action_mask,
        )
        current_states = [int(item["current_state"]) for item in batch]
        images = [
            qwen_visual_image_from_tokens(
                self.image_root,
                [int(token) for token in item["global_input_ids"].tolist()],
            )
            for item in batch
        ]
        text = [
            self._prompt(item["meta"], current_state)
            for item, current_state in zip(batch, current_states)
        ]
        qwen_inputs = self.processor(
            text=text,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        collated["qwen_input_ids"] = qwen_inputs["input_ids"]
        collated["qwen_attention_mask"] = qwen_inputs["attention_mask"]
        if "pixel_values" not in qwen_inputs or "image_grid_thw" not in qwen_inputs:
            raise ValueError("Qwen processor did not return pixel_values/image_grid_thw")
        collated["qwen_pixel_values"] = qwen_inputs["pixel_values"]
        collated["qwen_image_grid_thw"] = qwen_inputs["image_grid_thw"]
        return collated


class SFTRandomLocalActionDataset(Dataset):

    def __init__(
        self,
        filepath: str,
        require_action_label: bool = True,
        optimal_bias_alpha: float = 0.7,
    ):
        self.filepath = str(filepath)
        self.require_action_label = bool(require_action_label)
        self.optimal_bias_alpha = float(optimal_bias_alpha)
        raw_samples = SFTOptimLocalActionDataset._read_raw_samples(filepath)
        token_state_index, start_tokens_index = SFTOptimLocalActionDataset._build_indices(
            raw_samples,
            strict_token_state=self.require_action_label,
        )
        self.samples = self._build_samples(
            raw_samples=raw_samples,
            token_state_index=token_state_index,
            start_tokens_index=start_tokens_index,
        )
        if not self.samples:
            raise ValueError(f"No usable SFT-Random samples were built from {filepath}")
        self.action_names = action_names_for_meta(self.samples[0].meta)
        for sample in self.samples:
            sample_action_names = action_names_for_meta(sample.meta)
            if sample_action_names != self.action_names:
                raise ValueError(
                    "SFT-Random dataset mixes incompatible action spaces: "
                    f"{self.action_names} and {sample_action_names}"
                )

    def _build_samples(
        self,
        raw_samples: Sequence[Mapping[str, Any]],
        token_state_index: Mapping[Tuple[Any, Tuple[int, ...]], int],
        start_tokens_index: Mapping[Tuple[Any, int], Tuple[int, ...]],
    ) -> List[LocalActionSample]:
        samples: List[LocalActionSample] = []
        skipped = 0
        skipped_reasons: Dict[str, int] = {}
        mask_filtered = 0
        step_counters: Dict[Tuple[Any, ...], int] = {}
        history_cache: Dict[Tuple[Any, int], Optional[Tuple[int, ...]]] = {}

        for idx, obj in enumerate(raw_samples):
            meta = obj["meta"]
            group = sample_group_key(meta)
            level = int(meta["level"])
            height = int(meta.get("height", level))
            width = int(meta.get("width", level))
            current_state = normalize_state(obj["input_state"], meta)
            state_step = ordered_step_index(step_counters, meta, current_state)
            output_tokens = tuple(int(t) for t in obj["output_tokens"])

            if not self.require_action_label:
                mask = sft_random_action_mask(
                    meta,
                    current_state,
                    optimal_bias_alpha=self.optimal_bias_alpha,
                )
                enabled_actions = [
                    action_id for action_id, enabled in enumerate(mask) if enabled
                ]
                if not enabled_actions:
                    mask_filtered += 1
                    continue
                action_id = enabled_actions[0]
                next_state = current_state
                action_source = "soft_mask_only"
            else:
                action_id: Optional[int] = None
                next_state: Optional[int] = None
                action_source: str
                if is_maze_flip_meta(meta) and "action_id" in obj and "output_state" in obj:
                    action_id = int(obj["action_id"])
                    next_state = normalize_state(obj["output_state"], meta)
                    world_action_id = infer_task_action_id_from_states(
                        meta, current_state, next_state
                    )
                    expected_action_id = maze_flip_world_action_to_view(
                        meta, current_state, world_action_id
                    )
                    if action_id != expected_action_id:
                        raise ValueError(
                            "MazeFlip explicit action_id disagrees with world transition: "
                            f"stored={action_id}, expected={expected_action_id}"
                        )
                    action_source = "maze_flip_explicit_action_id"
                recovered = (
                    None
                    if action_id is not None
                    else recover_minibehaviour_action_from_output_state(
                        obj, current_state
                    )
                )
                if recovered is not None:
                    action_id, next_state = recovered
                    action_source = "output_state"
                else:

                    matched_next_state = (
                        None
                        if action_id is not None
                        else token_state_index.get((group, output_tokens))
                    )
                    if matched_next_state is not None:
                        next_state = matched_next_state
                        action_id = infer_task_action_id_from_states(
                            meta, current_state, next_state
                        )
                        action_source = "token_match"
                    elif action_id is None:

                        terminal_result = infer_terminal_action(meta, current_state)
                        if terminal_result is not None:
                            action_id, next_state = terminal_result
                            action_source = "terminal_recovery"
                        else:

                            skipped += 1
                            terminal_candidates = []
                            for candidate_action_id in range(len(action_names_for_meta(meta))):
                                result = transition_step(meta, current_state, candidate_action_id)
                                if result.valid and result.terminal:
                                    terminal_candidates.append(result.reason)
                            if len(terminal_candidates) > 1:
                                reason_key = "multiple_terminal_neighbors:" + ",".join(sorted(terminal_candidates))
                            elif len(terminal_candidates) == 0:
                                reason_key = "no_terminal_neighbor"
                            else:
                                reason_key = "unrecovered_single_terminal_neighbor"
                            skipped_reasons[reason_key] = skipped_reasons.get(reason_key, 0) + 1
                            continue

            start_state = (
                task_state_from_position(meta, meta["start_pos"], False)
                if is_minibehaviour_meta(meta)
                else int(meta.get("start_pos", current_state))
            )
            global_tokens = start_tokens_index.get((group, start_state))
            if global_tokens is None:
                if self.require_action_label:
                    raise ValueError(
                        f"Could not find global start tokens for start_state={start_state} "
                        f"inside group {group}"
                    )
                global_tokens = tuple(int(token) for token in obj["input_tokens"])
                action_source = "soft_mask_only_current_prompt"

            history_key = (group, current_state)
            if history_key not in history_cache:
                try:
                    history_cache[history_key] = model_action_history(meta, current_state)
                except ValueError:
                    history_cache[history_key] = None
            history = history_cache[history_key]
            if history is None:
                skipped += 1
                reason_key = "unreachable_from_episode_start"
                skipped_reasons[reason_key] = skipped_reasons.get(reason_key, 0) + 1
                continue
            sample = LocalActionSample(
                global_tokens=global_tokens,
                current_position=state_to_task_position(meta, current_state),
                map_size=(height, width),
                target_action_id=action_id,
                current_state=current_state,
                next_state=next_state,
                state_step=state_step,
                state_flags=(
                    (float(state_carrying(meta, current_state)),)
                    if is_minibehaviour_meta(meta)
                    else tuple()
                ),
                action_source=action_source,
                meta=meta,
                state_action_history=history,
            )
            if self.require_action_label:
                mask = sft_random_action_mask(
                    meta,
                    current_state,
                    optimal_bias_alpha=self.optimal_bias_alpha,
                )
                if sum(mask) <= 0:
                    mask_filtered += 1
                    continue
            samples.append(sample)

        if skipped > 0:
            log.warning(
                "SFTRandomLocalActionDataset: skipped %d / %d samples "
                "with unrecoverable action labels: %s",
                skipped,
                len(raw_samples),
                skipped_reasons,
            )
        if mask_filtered > 0:
            log.warning(
                "SFTRandomLocalActionDataset: filtered %d samples with empty "
                "target_closer_not_closer_uniform masks",
                mask_filtered,
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]


        if is_maze_flip_meta(sample.meta):
            model_position = torch.zeros(2, dtype=torch.float32)
            model_map_size = torch.ones(2, dtype=torch.float32)
            model_state_step = torch.tensor(0.0, dtype=torch.float32)
        else:
            model_position = torch.tensor(sample.current_position, dtype=torch.float32)
            model_map_size = torch.tensor(sample.map_size, dtype=torch.float32)
            model_state_step = torch.tensor(sample.state_step, dtype=torch.float32)
        item = {
            "global_input_ids": torch.tensor(sample.global_tokens, dtype=torch.long),
            "position": model_position,
            "map_size": model_map_size,
            "labels": torch.tensor(sample.target_action_id, dtype=torch.long),
            "state_step": model_state_step,
            "current_state": sample.current_state,
            "next_state": sample.next_state,
            "action_source": sample.action_source,
            "state_action_history": torch.tensor(
                sample.state_action_history, dtype=torch.long
            ),
            "valid_action_mask": torch.tensor(
                sft_random_action_mask(
                    sample.meta,
                    sample.current_state,
                    optimal_bias_alpha=self.optimal_bias_alpha,
                ),
                dtype=torch.float32,
            ),
            "meta": sample.meta,
        }
        if sample.state_flags:
            item["state_flags"] = torch.tensor(sample.state_flags, dtype=torch.float32)
        return item


def fourier_position_features(
    position: torch.Tensor,
    map_size: torch.Tensor,
    num_frequencies: int,
) -> torch.Tensor:
    if num_frequencies <= 0:
        raise ValueError("num_frequencies must be positive")
    if position.dim() != 2 or position.size(-1) != 2:
        raise ValueError("position must have shape (batch, 2)")
    if map_size.dim() != 2 or map_size.size(-1) != 2:
        raise ValueError("map_size must have shape (batch, 2)")

    position = position.to(dtype=torch.float32)
    map_size = map_size.to(device=position.device, dtype=torch.float32)
    if torch.any(map_size <= 0):
        raise ValueError("map_size values must be positive")

    rows, cols = position[:, 0], position[:, 1]
    heights, widths = map_size[:, 0], map_size[:, 1]
    if torch.any(rows < 0) or torch.any(cols < 0):
        raise ValueError("position values must be non-negative")
    if torch.any(rows >= heights) or torch.any(cols >= widths):
        raise ValueError("position must be inside map_size")

    u = (rows + 0.5) / heights
    v = (cols + 0.5) / widths
    frequencies = (2 ** torch.arange(num_frequencies, device=position.device)).float()
    angles = frequencies * math.pi

    u_angles = u.unsqueeze(1) * angles.unsqueeze(0)
    v_angles = v.unsqueeze(1) * angles.unsqueeze(0)
    u_features = torch.stack((torch.sin(u_angles), torch.cos(u_angles)), dim=-1).flatten(1)
    v_features = torch.stack((torch.sin(v_angles), torch.cos(v_angles)), dim=-1).flatten(1)
    size_features = torch.stack((torch.log(heights), torch.log(widths)), dim=-1)
    return torch.cat((u_features, v_features, size_features), dim=-1)


class FourierPositionEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_frequencies: int = 8,
        mlp_hidden_size: int = 512,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_frequencies = int(num_frequencies)
        self.mlp_hidden_size = int(mlp_hidden_size)
        input_size = 4 * self.num_frequencies + 2
        self.mlp = nn.Sequential(
            nn.Linear(input_size, self.mlp_hidden_size),
            nn.SiLU(),
            nn.Linear(self.mlp_hidden_size, self.hidden_size),
        )

    def forward(self, position: torch.Tensor, map_size: torch.Tensor) -> torch.Tensor:
        features = fourier_position_features(
            position=position,
            map_size=map_size,
            num_frequencies=self.num_frequencies,
        )
        weight = self.mlp[0].weight
        return self.mlp(features.to(device=weight.device, dtype=weight.dtype))

    def to_config(self) -> Dict[str, int]:
        return {
            "hidden_size": self.hidden_size,
            "num_frequencies": self.num_frequencies,
            "mlp_hidden_size": self.mlp_hidden_size,
        }


class ImplicitStateEncoder(nn.Module):


    def __init__(self, hidden_size: int, num_actions: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_actions = int(num_actions)
        self.state_query = nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.state_query, mean=0.0, std=self.hidden_size ** -0.5)
        self.context_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.action_embedding = nn.Embedding(self.num_actions, self.hidden_size)
        self.transition = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.state_norm = nn.LayerNorm(self.hidden_size)

    def initial_state(self, context_embedding: torch.Tensor) -> torch.Tensor:
        if context_embedding.dim() != 2 or context_embedding.size(-1) != self.hidden_size:
            raise ValueError("context_embedding must have shape (batch, hidden_size)")
        query = self.state_query.unsqueeze(0).expand(context_embedding.size(0), -1)
        return self.state_norm(query + self.context_projection(context_embedding))

    def update(self, state: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        action_ids = action_ids.to(device=state.device, dtype=torch.long)
        action_embedding = self.action_embedding(action_ids)
        return self.state_norm(self.transition(action_embedding, state))

    def forward(
        self,
        context_embedding: torch.Tensor,
        action_history: Optional[torch.Tensor] = None,
        action_history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        state = self.initial_state(context_embedding)
        if action_history is None or action_history.size(1) == 0:
            return state
        if action_history_mask is None:
            action_history_mask = torch.ones_like(action_history, dtype=torch.bool)
        for step in range(action_history.size(1)):
            updated = self.update(state, action_history[:, step])
            active = action_history_mask[:, step].to(device=state.device).unsqueeze(-1)
            state = torch.where(active, updated, state)
        return state

    def to_config(self) -> Dict[str, Any]:
        return {
            "type": "recurrent_implicit_state",
            "hidden_size": self.hidden_size,
            "num_actions": self.num_actions,
        }


class ImplicitQueryStateEncoder(nn.Module):


    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.state_query = nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.state_query, mean=0.0, std=self.hidden_size ** -0.5)

    def initial_state(self, context_embedding: torch.Tensor) -> torch.Tensor:
        return self.state_query.unsqueeze(0).expand(context_embedding.size(0), -1)

    def update(self, state: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        del action_ids
        return state

    def forward(
        self,
        context_embedding: torch.Tensor,
        action_history: Optional[torch.Tensor] = None,
        action_history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del action_history, action_history_mask
        return self.initial_state(context_embedding)

    def to_config(self) -> Dict[str, Any]:
        return {"type": "implicit_query", "hidden_size": self.hidden_size}


class LocalActionPlanningModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_frequencies: int = 8,
        position_mlp_hidden_size: int = 512,
        global_start_token_id: int = GLOBAL_START_TOKEN_ID,
        global_end_token_id: int = GLOBAL_END_TOKEN_ID,
        action_names: Sequence[str] = ACTION_NAMES,
        num_state_flags: Optional[int] = None,
        action_token_ids: Optional[Sequence[int]] = None,
        state_encoder_type: str = "recurrent_implicit_state",
        action_output_mode: str = "text_token",
        ablation_modules: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.global_start_token_id = int(global_start_token_id)
        self.global_end_token_id = int(global_end_token_id)
        self.action_names = tuple(str(name) for name in action_names)
        self.state_flag_fusion_mode = "add"
        self.state_token_mode = "append"
        if action_output_mode not in ("text_token", "linear"):
            raise ValueError("action_output_mode must be text_token or linear")
        self.action_output_mode = action_output_mode
        self.ablation_modules = bool(ablation_modules or action_output_mode == "linear")


        self.num_state_flags = int(num_state_flags or 0)
        hidden_size = backbone_hidden_size(backbone)
        if state_encoder_type == "recurrent_implicit_state":
            self.state_encoder = ImplicitStateEncoder(
                hidden_size=hidden_size,
                num_actions=len(self.action_names),
            )
        elif state_encoder_type == "implicit_query":
            self.state_encoder = ImplicitQueryStateEncoder(hidden_size=hidden_size)
        else:
            raise ValueError(f"Unsupported implicit state encoder: {state_encoder_type}")
        if action_token_ids is None:
            raise ValueError("action_token_ids is required")
        if len(action_token_ids) != len(self.action_names):
            raise ValueError("action_token_ids length must match action_names length")
        self.register_buffer(
            "action_token_ids",
            torch.tensor([int(token_id) for token_id in action_token_ids], dtype=torch.long),
            persistent=False,
        )
        self.state_flag_encoder: Optional[nn.Linear]
        if self.num_state_flags > 0:
            cpu_rng_state = torch.get_rng_state()
            self.state_flag_encoder = nn.Linear(self.num_state_flags, hidden_size, bias=False)
            nn.init.zeros_(self.state_flag_encoder.weight)
            torch.set_rng_state(cpu_rng_state)
        else:
            self.state_flag_encoder = None
        self.action_head = nn.Linear(hidden_size, len(self.action_names)) if self.ablation_modules else None
        if self.action_head is not None:
            self.action_head.requires_grad_(self.action_output_mode == "linear")
        self._last_loss_metrics: Dict[str, float] = {}

    def encode_state(
        self,
        position: torch.Tensor,
        map_size: torch.Tensor,
        state_flags: Optional[torch.Tensor],
        state_step: Optional[torch.Tensor] = None,
        global_input_ids: Optional[torch.Tensor] = None,
        state_action_history: Optional[torch.Tensor] = None,
        state_action_history_mask: Optional[torch.Tensor] = None,
        latent_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latent_state is not None:
            state_embedding = latent_state
        else:
            if global_input_ids is None:
                raise ValueError("global_input_ids is required to initialize implicit state")
            token_embeddings = self.backbone.get_input_embeddings()(
                global_input_ids.to(next(self.state_encoder.parameters()).device)
            )
            context_embedding = token_embeddings.float().mean(dim=1).to(
                dtype=self.state_encoder.state_query.dtype
            )
            state_embedding = self.state_encoder(
                context_embedding,
                state_action_history,
                state_action_history_mask,
            )
        if self.num_state_flags <= 0:
            return state_embedding
        if state_flags is None:
            state_flags = torch.zeros(
                (position.size(0), self.num_state_flags),
                dtype=torch.float32,
                device=position.device,
            )
        if state_flags.shape != (position.size(0), self.num_state_flags):
            raise ValueError(
                f"state_flags must have shape {(position.size(0), self.num_state_flags)}, "
                f"got {tuple(state_flags.shape)}"
            )
        if self.state_flag_encoder is None:
            raise ValueError("state_flag_encoder is missing despite num_state_flags > 0")
        flag_param = next(self.state_flag_encoder.parameters())
        state_flags = state_flags.to(device=flag_param.device, dtype=flag_param.dtype)
        return state_embedding + self.state_flag_encoder(state_flags)

    def build_global_sequence(self, global_input_ids: torch.Tensor) -> torch.Tensor:
        if global_input_ids.dim() != 2:
            raise ValueError("global_input_ids must have shape (batch, seq_len)")
        batch_size = global_input_ids.size(0)
        start = torch.full(
            (batch_size, 1),
            self.global_start_token_id,
            dtype=global_input_ids.dtype,
            device=global_input_ids.device,
        )
        end = torch.full(
            (batch_size, 1),
            self.global_end_token_id,
            dtype=global_input_ids.dtype,
            device=global_input_ids.device,
        )
        return torch.cat((start, global_input_ids, end), dim=1)

    def _action_logits_from_hidden(self, last_hidden: torch.Tensor) -> torch.Tensor:
        if self.action_output_mode == "linear":
            return self.action_head(last_hidden.to(self.action_head.weight.dtype))
        output_embeddings = None
        getter = getattr(self.backbone, "get_output_embeddings", None)
        if callable(getter):
            output_embeddings = getter()
        if output_embeddings is None:
            qwen = self._qwen_module()
            getter = getattr(qwen, "get_output_embeddings", None)
            if callable(getter):
                output_embeddings = getter()
            elif hasattr(qwen, "lm_head"):
                output_embeddings = qwen.lm_head
        if output_embeddings is None:
            raise ValueError("action_output_mode='text_token' requires LM output embeddings")
        output_weight = getattr(output_embeddings, "weight", None)
        output_dtype = output_weight.dtype if output_weight is not None else last_hidden.dtype
        vocab_logits = output_embeddings(last_hidden.to(output_dtype))
        token_ids = self.action_token_ids.to(device=vocab_logits.device)
        return vocab_logits.index_select(dim=-1, index=token_ids)

    def _fuse_state_embedding(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        state_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        state_embedding = state_embedding.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        ).unsqueeze(1)
        inputs_embeds = torch.cat((inputs_embeds, state_embedding), dim=1)
        state_attention = torch.ones(
            (attention_mask.size(0), 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat((attention_mask, state_attention), dim=1)
        readout_indices = torch.full(
            (attention_mask.size(0),),
            attention_mask.size(1) - 1,
            dtype=torch.long,
            device=attention_mask.device,
        )
        return inputs_embeds, attention_mask, readout_indices

    @staticmethod
    def _select_readout_hidden(
        hidden_states: torch.Tensor,
        readout_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if readout_indices is None:
            return hidden_states[:, -1, :]
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, readout_indices.to(device=hidden_states.device), :]

    def encode_global_cache(
        self,
        global_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, torch.Tensor]:
        global_sequence = self.build_global_sequence(global_input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(global_sequence, dtype=torch.long)
        else:
            prefix_suffix = torch.ones(
                (attention_mask.size(0), 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat((prefix_suffix, attention_mask, prefix_suffix), dim=1)
        outputs = self.backbone(
            input_ids=global_sequence,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        return outputs.past_key_values, attention_mask

    def forward_with_cache(
        self,
        position: torch.Tensor,
        map_size: torch.Tensor,
        past_key_values: Any,
        global_attention_mask: torch.Tensor,
        state_flags: Optional[torch.Tensor] = None,
        state_step: Optional[torch.Tensor] = None,
        latent_state: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        state_embedding = self.encode_state(
            position, map_size, state_flags, state_step=state_step, latent_state=latent_state
        )
        token_dtype = self.backbone.get_input_embeddings().weight.dtype
        state_embedding = state_embedding.to(
            device=global_attention_mask.device,
            dtype=token_dtype,
        ).unsqueeze(1)
        state_attention = torch.ones(
            (global_attention_mask.size(0), 1),
            dtype=global_attention_mask.dtype,
            device=global_attention_mask.device,
        )
        attention_mask = torch.cat((global_attention_mask, state_attention), dim=1)
        last_hidden = self._last_hidden_state(
            inputs_embeds=state_embedding,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = self._action_logits_from_hidden(last_hidden)
        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def forward(
        self,
        global_input_ids: torch.Tensor,
        position: torch.Tensor,
        map_size: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        valid_action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        state_flags: Optional[torch.Tensor] = None,
        qwen_input_ids: Optional[torch.Tensor] = None,
        qwen_attention_mask: Optional[torch.Tensor] = None,
        qwen_pixel_values: Optional[torch.Tensor] = None,
        qwen_image_grid_thw: Optional[torch.Tensor] = None,
        state_step: Optional[torch.Tensor] = None,
        state_action_history: Optional[torch.Tensor] = None,
        state_action_history_mask: Optional[torch.Tensor] = None,
        latent_state: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> SequenceClassifierOutput:
        state_embedding = self.encode_state(
            position,
            map_size,
            state_flags,
            state_step=state_step,
            global_input_ids=global_input_ids,
            state_action_history=state_action_history,
            state_action_history_mask=state_action_history_mask,
            latent_state=latent_state,
        )
        if qwen_input_ids is not None:
            last_hidden = self._qwen_visual_last_hidden_state(
                qwen_input_ids=qwen_input_ids,
                qwen_attention_mask=qwen_attention_mask,
                qwen_pixel_values=qwen_pixel_values,
                qwen_image_grid_thw=qwen_image_grid_thw,
                state_embedding=state_embedding,
            )
        else:
            global_sequence = self.build_global_sequence(global_input_ids)
            token_embeddings = self.backbone.get_input_embeddings()(global_sequence)

            if attention_mask is None:
                attention_mask = torch.ones(
                    global_sequence.shape,
                    dtype=torch.long,
                    device=global_sequence.device,
                )
            else:
                prefix_suffix = torch.ones(
                    (attention_mask.size(0), 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat((prefix_suffix, attention_mask, prefix_suffix), dim=1)
            inputs_embeds, attention_mask, readout_indices = self._fuse_state_embedding(
                token_embeddings,
                attention_mask,
                state_embedding,
            )

            last_hidden = self._last_hidden_state(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
                readout_indices=readout_indices,
            )
        logits = self._action_logits_from_hidden(last_hidden)

        loss = None
        metrics: Dict[str, torch.Tensor] = {}
        if valid_action_mask is not None:
            mask = valid_action_mask.to(device=logits.device, dtype=torch.float32)
            if mask.shape != logits.shape:
                raise ValueError(
                    f"valid_action_mask shape {tuple(mask.shape)} must match logits shape {tuple(logits.shape)}"
                )
            valid_counts = mask.sum(dim=-1, keepdim=True)
            if torch.any(valid_counts <= 0):
                raise ValueError("valid_action_mask must contain at least one valid action per sample")
            target_distribution = mask / valid_counts
            log_probs = F.log_softmax(logits.float(), dim=-1)
            target_loss = -(target_distribution * log_probs).sum(dim=-1).mean()
            loss = target_loss
            metrics["target_ce_loss"] = target_loss.detach()
        elif labels is not None:
            label_loss = F.cross_entropy(logits.float(), labels.long())
            loss = label_loss
            metrics["target_ce_loss"] = label_loss.detach()

        self._last_loss_metrics = {
            key: float(value.detach().float().cpu())
            for key, value in metrics.items()
        }

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def _qwen_module(self) -> nn.Module:
        module: nn.Module = self.backbone
        base_model = getattr(module, "base_model", None)
        if base_model is not None and hasattr(base_model, "model"):
            return base_model.model
        return module

    def _qwen_visual_last_hidden_state(
        self,
        qwen_input_ids: torch.Tensor,
        qwen_attention_mask: Optional[torch.Tensor],
        qwen_pixel_values: Optional[torch.Tensor],
        qwen_image_grid_thw: Optional[torch.Tensor],
        state_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if qwen_pixel_values is None or qwen_image_grid_thw is None:
            raise ValueError("Qwen visual forward requires pixel_values and image_grid_thw")
        qwen = self._qwen_module()
        if not all(hasattr(qwen, attr) for attr in ("model", "visual", "get_rope_index")):
            raise ValueError("qwen_* visual inputs require a Qwen2.5-VL backbone")
        input_ids = qwen_input_ids.to(device=state_embedding.device)
        attention_mask = (
            torch.ones_like(input_ids, dtype=torch.long)
            if qwen_attention_mask is None
            else qwen_attention_mask.to(device=input_ids.device)
        )
        inputs_embeds = qwen.model.embed_tokens(input_ids)
        pixel_values = qwen_pixel_values.to(device=input_ids.device).type(qwen.visual.dtype)
        image_grid_thw = qwen_image_grid_thw.to(device=input_ids.device)
        image_embeds = qwen.visual(pixel_values, grid_thw=image_grid_thw)
        image_token_id = int(qwen.config.image_token_id)
        image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        n_image_tokens = int((input_ids == image_token_id).sum().item())
        if n_image_tokens != int(image_embeds.shape[0]):
            raise ValueError(
                "Qwen image features and image tokens do not match: "
                f"tokens={n_image_tokens}, features={int(image_embeds.shape[0])}"
            )
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask.to(inputs_embeds.device),
            image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )
        position_ids, _rope_deltas = qwen.get_rope_index(
            input_ids,
            image_grid_thw,
            None,
            None,
            attention_mask,
        )
        inputs_embeds, attention_mask, readout_indices = self._fuse_state_embedding(
            inputs_embeds,
            attention_mask,
            state_embedding,
        )
        valid_lengths = attention_mask[:, :-1].long().sum(dim=1).clamp_min(1)
        last_indices = (valid_lengths - 1).to(device=position_ids.device)
        batch_indices = torch.arange(position_ids.size(1), device=position_ids.device)
        state_position = position_ids[:, batch_indices, last_indices] + 1
        position_ids = torch.cat((position_ids, state_position.unsqueeze(-1)), dim=-1)

        outputs = qwen.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        if hasattr(outputs, "last_hidden_state"):
            return self._select_readout_hidden(outputs.last_hidden_state, readout_indices)
        return self._select_readout_hidden(outputs[0], readout_indices)

    @torch.no_grad()
    def encode_qwen_visual_cache(
        self,
        qwen_input_ids: torch.Tensor,
        qwen_attention_mask: Optional[torch.Tensor],
        qwen_pixel_values: torch.Tensor,
        qwen_image_grid_thw: torch.Tensor,
    ) -> Dict[str, Any]:
        qwen = self._qwen_module()
        input_ids = qwen_input_ids
        attention_mask = (
            torch.ones_like(input_ids, dtype=torch.long)
            if qwen_attention_mask is None else qwen_attention_mask
        )
        inputs_embeds = qwen.model.embed_tokens(input_ids)
        image_embeds = qwen.visual(
            qwen_pixel_values.type(qwen.visual.dtype), grid_thw=qwen_image_grid_thw
        )
        image_mask = (input_ids == int(qwen.config.image_token_id)).unsqueeze(-1).expand_as(inputs_embeds)
        if int((input_ids == int(qwen.config.image_token_id)).sum().item()) != int(image_embeds.shape[0]):
            raise ValueError("Qwen image features and image tokens do not match")
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask, image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        )
        position_ids, _ = qwen.get_rope_index(
            input_ids, qwen_image_grid_thw, None, None, attention_mask
        )
        outputs = qwen.model(
            input_ids=None, inputs_embeds=inputs_embeds, position_ids=position_ids,
            attention_mask=attention_mask, past_key_values=None, use_cache=True,
            output_attentions=False, output_hidden_states=False, return_dict=True,
        )
        valid_lengths = attention_mask.long().sum(dim=1).clamp_min(1)
        batch_indices = torch.arange(position_ids.size(1), device=position_ids.device)
        last_indices = (valid_lengths - 1).to(device=position_ids.device)
        return {
            "past_key_values": outputs.past_key_values,
            "prefix_attention_mask": attention_mask,
            "last_position": position_ids[:, batch_indices, last_indices],
            "prefix_length": int(attention_mask.size(1)),
        }

    @torch.no_grad()
    def forward_with_qwen_visual_cache(
        self,
        position: torch.Tensor,
        map_size: torch.Tensor,
        prefix_cache: Mapping[str, Any],
        latent_state: torch.Tensor,
        state_flags: Optional[torch.Tensor] = None,
        state_step: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        qwen = self._qwen_module()
        state_embedding = self.encode_state(
            position, map_size, state_flags, state_step=state_step, latent_state=latent_state
        ).to(dtype=qwen.model.embed_tokens.weight.dtype).unsqueeze(1)
        attention_mask = prefix_cache["prefix_attention_mask"]
        state_attention = torch.ones(
            (attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device
        )
        cache = prefix_cache["past_key_values"]
        try:
            outputs = qwen.model(
                input_ids=None, inputs_embeds=state_embedding,
                position_ids=(prefix_cache["last_position"] + 1).unsqueeze(-1),
                attention_mask=torch.cat((attention_mask, state_attention), dim=1),
                past_key_values=cache, use_cache=False, output_attentions=False,
                output_hidden_states=False, return_dict=True,
                cache_position=torch.tensor(
                    [prefix_cache["prefix_length"]], dtype=torch.long, device=position.device
                ),
            )
        finally:
            if hasattr(cache, "crop"):
                cache.crop(prefix_cache["prefix_length"])
        return SequenceClassifierOutput(
            logits=self._action_logits_from_hidden(outputs.last_hidden_state[:, -1, :])
        )

    def _last_hidden_state(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
        use_cache: bool,
        readout_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for decoder in self._decoder_candidates():
            try:
                outputs = decoder(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=use_cache,
                )
                if hasattr(outputs, "last_hidden_state"):
                    return self._select_readout_hidden(outputs.last_hidden_state, readout_indices)
            except Exception:
                continue

        outputs = self.backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=True,
            use_cache=use_cache,
        )
        return self._select_readout_hidden(outputs.hidden_states[-1], readout_indices)

    def _decoder_candidates(self) -> List[nn.Module]:
        candidates: List[nn.Module] = []

        def add_decoder_from(module: Any) -> None:
            decoder_getter = getattr(module, "get_decoder", None)
            if callable(decoder_getter):
                try:
                    decoder = decoder_getter()
                except Exception:
                    decoder = None
                if isinstance(decoder, nn.Module) and all(decoder is not item for item in candidates):
                    candidates.append(decoder)

        add_decoder_from(self.backbone)

        base_getter = getattr(self.backbone, "get_base_model", None)
        if callable(base_getter):
            try:
                add_decoder_from(base_getter())
            except Exception:
                pass

        module: Any = self.backbone
        for attr in ("base_model", "model", "model"):
            module = getattr(module, attr, None)
            if module is None:
                break
            add_decoder_from(module)
            if isinstance(module, nn.Module) and module.__class__.__name__.endswith("Model"):
                if all(module is not item for item in candidates):
                    candidates.append(module)

        return candidates

    @torch.no_grad()
    def initialize_latent_state(
        self, global_input_ids: torch.Tensor,
        action_history: Optional[torch.Tensor] = None,
        action_history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_embeddings = self.backbone.get_input_embeddings()(global_input_ids)
        context_embedding = token_embeddings.float().mean(dim=1).to(
            dtype=self.state_encoder.state_query.dtype
        )
        return self.state_encoder(context_embedding, action_history, action_history_mask)

    @torch.no_grad()
    def update_latent_state(
        self, latent_state: torch.Tensor, action_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.state_encoder.update(latent_state, action_ids)

    @torch.no_grad()
    def predict_action(
        self,
        global_input_ids: torch.Tensor,
        position: torch.Tensor,
        map_size: torch.Tensor,
        use_cache: bool = True,
        state_flags: Optional[torch.Tensor] = None,
        state_step: Optional[torch.Tensor] = None,
        latent_state: Optional[torch.Tensor] = None,
        qwen_prefix_cache: Optional[Mapping[str, Any]] = None,
        **forward_kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        if latent_state is None:
            latent_state = self.initialize_latent_state(
                global_input_ids,
                forward_kwargs.get("state_action_history"),
                forward_kwargs.get("state_action_history_mask"),
            )
        if use_cache and (qwen_prefix_cache is not None or forward_kwargs.get("qwen_input_ids") is not None):
            if qwen_prefix_cache is None:
                qwen_prefix_cache = self.encode_qwen_visual_cache(
                    qwen_input_ids=forward_kwargs["qwen_input_ids"],
                    qwen_attention_mask=forward_kwargs.get("qwen_attention_mask"),
                    qwen_pixel_values=forward_kwargs["qwen_pixel_values"],
                    qwen_image_grid_thw=forward_kwargs["qwen_image_grid_thw"],
                )
            outputs = self.forward_with_qwen_visual_cache(
                position=position, map_size=map_size, prefix_cache=qwen_prefix_cache,
                latent_state=latent_state, state_flags=state_flags, state_step=state_step,
            )
        elif use_cache and not forward_kwargs:
            past_key_values, global_attention_mask = self.encode_global_cache(global_input_ids)
            outputs = self.forward_with_cache(
                position=position,
                map_size=map_size,
                past_key_values=past_key_values,
                global_attention_mask=global_attention_mask,
                state_flags=state_flags,
                state_step=state_step,
                latent_state=latent_state,
            )
        else:
            outputs = self(
                global_input_ids=global_input_ids,
                position=position,
                map_size=map_size,
                state_flags=state_flags,
                state_step=state_step,
                latent_state=latent_state,
                **forward_kwargs,
            )
        action_ids = torch.argmax(outputs.logits, dim=-1)
        return action_ids, outputs.logits

    def save_local_components(self, save_directory: str) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        state = {
            "state_encoder": self.state_encoder.state_dict(),
        }
        if self.state_flag_encoder is not None:
            state["state_flag_encoder"] = self.state_flag_encoder.state_dict()
        if self.action_head is not None:
            state["action_head"] = self.action_head.state_dict()
        torch.save(state, save_path / "local_action_components.pt")
        with open(save_path / "local_action_config.json", "w", encoding="utf-8") as f:
            config = {
                "state_encoder": self.state_encoder.to_config(),
                "actions": list(self.action_names),
                "global_start_token_id": self.global_start_token_id,
                "global_end_token_id": self.global_end_token_id,
                "state_token_mode": "append",
                "action_output_mode": self.action_output_mode,
                "ablation_modules": self.ablation_modules,
            }
            config["action_token_ids"] = [int(x) for x in self.action_token_ids.cpu().tolist()]
            if self.num_state_flags > 0:
                config["num_state_flags"] = self.num_state_flags
                config["state_flag_fusion_mode"] = self.state_flag_fusion_mode
            json.dump(
                config,
                f,
                indent=2,
            )


def set_backbone_trainable(model: LocalActionPlanningModel, trainable: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = trainable


def save_local_action_model(model: LocalActionPlanningModel, save_directory: str) -> None:
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_local_components(str(save_path))

    if hasattr(model.backbone, "save_pretrained") and hasattr(model.backbone, "peft_config"):
        backbone_dir = save_path / "backbone"
        model.backbone.save_pretrained(str(backbone_dir))


def load_local_action_components(
    model: LocalActionPlanningModel,
    checkpoint_directory: str,
    map_location: str = "cpu",
) -> None:
    checkpoint_path = Path(checkpoint_directory) / "local_action_components.pt"
    if not checkpoint_path.exists():
        raise ValueError(f"Missing local action checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if "state_encoder" not in state:
        raise ValueError(
            "This checkpoint uses the legacy explicit position state encoder. "
            "Implicit-state models must be trained from a new checkpoint."
        )
    model.state_encoder.load_state_dict(state["state_encoder"])
    for name in ("action_head",):
        module = getattr(model, name)
        if module is not None:
            if name not in state:
                raise ValueError(f"Ablation checkpoint is missing {name} weights")
            module.load_state_dict(state[name])
        elif name in state:
            raise ValueError(f"Checkpoint has {name}, but model was built without ablation modules")
    if "state_flag_encoder" in state:
        if model.state_flag_encoder is None:
            raise ValueError(
                "Checkpoint contains state_flag_encoder, but model was constructed "
                "with num_state_flags=0"
            )
        model.state_flag_encoder.load_state_dict(state["state_flag_encoder"])
    elif model.state_flag_encoder is not None:
        nn.init.zeros_(model.state_flag_encoder.weight)


def read_local_action_config(checkpoint_directory: str) -> Dict[str, Any]:
    config_path = Path(checkpoint_directory) / "local_action_config.json"
    if not config_path.exists():
        raise ValueError(f"Missing local action config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_local_action_model(
    backbone: nn.Module,
    checkpoint_directory: str,
    map_location: str = "cpu",
) -> LocalActionPlanningModel:
    config = read_local_action_config(checkpoint_directory)
    state_config = config.get("state_encoder")
    state_encoder_type = state_config.get("type") if state_config else None
    if state_encoder_type not in {"recurrent_implicit_state", "implicit_query"}:
        raise ValueError(
            "Checkpoint does not contain a supported implicit state encoder; "
            "legacy position-state checkpoints are not architecture-compatible."
        )
    model = LocalActionPlanningModel(
        backbone=backbone,
        global_start_token_id=config.get("global_start_token_id", GLOBAL_START_TOKEN_ID),
        global_end_token_id=config.get("global_end_token_id", GLOBAL_END_TOKEN_ID),
        action_names=config.get("actions", ACTION_NAMES),
        num_state_flags=config.get("num_state_flags"),
        action_token_ids=config.get("action_token_ids"),
        state_encoder_type=state_encoder_type,
        action_output_mode=config.get("action_output_mode", "text_token"),
        ablation_modules=config.get("ablation_modules", False),
    )
    load_local_action_components(model, checkpoint_directory, map_location=map_location)
    return model
