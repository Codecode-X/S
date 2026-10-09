from __future__ import annotations
import argparse
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

LOG = logging.getLogger("maze_flip_builder")
ACTION_NAMES: Tuple[str, ...] = ("UP", "DOWN", "LEFT", "RIGHT")
ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}
DELTA_TO_ACTION_ID = {delta: action_id for (action_id, delta) in ACTION_DELTAS.items()}
DATASET_VERSION = "maze_flip_v2_implicit_view_state"
DEFAULT_TOKEN_GRID_SIZE = 16


def log_stage(message: str) -> None:
    LOG.info("========== [MAZE-FLIP] %s ==========", message)


def state_to_position(state: int, width: int) -> Tuple[int, int]:
    return divmod(int(state), int(width))


def infer_world_action_id(current_state: int, next_state: int, width: int) -> int:
    (current_row, current_col) = state_to_position(current_state, width)
    (next_row, next_col) = state_to_position(next_state, width)
    delta = (next_row - current_row, next_col - current_col)
    if delta not in DELTA_TO_ACTION_ID:
        raise ValueError(
            f"Maze transition must move to one adjacent cell: state={current_state}, next_state={next_state}, delta={delta}"
        )
    return DELTA_TO_ACTION_ID[delta]


def view_flip_for_state(start_state: int, state: int, width: int) -> Tuple[bool, bool]:
    (start_row, start_col) = state_to_position(start_state, width)
    (row, col) = state_to_position(state, width)
    return (bool((col - start_col) % 2), bool((row - start_row) % 2))


def view_transform_name(flip_vertical: bool, flip_horizontal: bool) -> str:
    if flip_vertical and flip_horizontal:
        return "flip_both"
    if flip_vertical:
        return "flip_vertical"
    if flip_horizontal:
        return "flip_horizontal"
    return "identity"


def remap_world_action_to_view(
    world_action_id: int, flip_vertical: bool, flip_horizontal: bool
) -> int:
    action_id = int(world_action_id)
    if action_id not in ACTION_DELTAS:
        raise ValueError(f"Unsupported Maze action id: {action_id}")
    if flip_vertical:
        if action_id == 0:
            action_id = 1
        elif action_id == 1:
            action_id = 0
    if flip_horizontal:
        if action_id == 2:
            action_id = 3
        elif action_id == 3:
            action_id = 2
    return action_id


def infer_token_grid_size(tokens: Sequence[int]) -> int:
    grid_size = math.isqrt(len(tokens))
    if grid_size * grid_size != len(tokens):
        raise ValueError(
            f"Visual token count must be a square grid, got {len(tokens)} tokens"
        )
    return grid_size


def flip_visual_tokens(
    tokens: Sequence[int], flip_vertical: bool, flip_horizontal: bool
) -> List[int]:
    grid_size = infer_token_grid_size(tokens)
    grid = [
        [int(value) for value in tokens[row * grid_size : (row + 1) * grid_size]]
        for row in range(grid_size)
    ]
    if flip_vertical:
        grid.reverse()
    if flip_horizontal:
        for row in grid:
            row.reverse()
    return [value for row in grid for value in row]


def maze_flip_meta(meta: MutableMapping[str, Any]) -> None:
    meta["task_variant"] = "maze_flip"
    meta["maze_flip_version"] = DATASET_VERSION
    meta["state_coordinate_frame"] = "world"
    meta["action_coordinate_frame"] = "current_view"
    meta["maze_flip_rule"] = {
        "left_right_world_move": "toggle_flip_vertical",
        "up_down_world_move": "toggle_flip_horizontal",
        "route_semantics": "world_route_unchanged",
        "visual_token_transform": "spatial_vq_grid_flip",
    }


def maze_dimensions(meta: Mapping[str, Any]) -> Tuple[int, int, int]:
    level = int(meta["level"])
    return (
        int(meta.get("height", level)),
        int(meta.get("width", level)),
        int(meta["start_pos"]),
    )


def transform_random_sample(sample: Mapping[str, Any]) -> Dict[str, Any]:
    transformed = dict(sample)
    transformed["meta"] = dict(sample["meta"])
    (_height, width, start_state) = maze_dimensions(transformed["meta"])
    current_state = int(sample["input_state"])
    next_state = int(sample["output_state"])
    world_action_id = infer_world_action_id(current_state, next_state, width)
    before = view_flip_for_state(start_state, current_state, width)
    after = view_flip_for_state(start_state, next_state, width)
    action_id = remap_world_action_to_view(world_action_id, *before)
    transformed["input_tokens"] = flip_visual_tokens(sample["input_tokens"], *before)
    transformed["output_tokens"] = flip_visual_tokens(sample["output_tokens"], *after)
    transformed["action_id"] = action_id
    transformed["action"] = ACTION_NAMES[action_id]
    transformed["world_action_id"] = world_action_id
    transformed["world_action"] = ACTION_NAMES[world_action_id]
    maze_flip_meta(transformed["meta"])
    return transformed


def _normalize_route(raw_state: Any) -> List[int]:
    if isinstance(raw_state, list):
        return [int(state) for state in raw_state]
    if isinstance(raw_state, tuple):
        return [int(state) for state in raw_state]
    return [int(raw_state)]


def transform_validation_sample(sample: Mapping[str, Any]) -> Dict[str, Any]:
    transformed = dict(sample)
    transformed["meta"] = dict(sample["meta"])
    (_height, width, start_state) = maze_dimensions(transformed["meta"])
    route = _normalize_route(sample["input_state"])
    world_action_ids: List[int] = []
    action_ids: List[int] = []
    for current_state, next_state in zip(route, route[1:]):
        world_action_id = infer_world_action_id(current_state, next_state, width)
        before = view_flip_for_state(start_state, current_state, width)
        world_action_ids.append(world_action_id)
        action_ids.append(remap_world_action_to_view(world_action_id, *before))
    first_flip = view_flip_for_state(start_state, route[0], width)
    final_flip = view_flip_for_state(start_state, route[-1], width)
    transformed["input_tokens"] = flip_visual_tokens(
        sample["input_tokens"], *first_flip
    )
    transformed["output_tokens"] = flip_visual_tokens(
        sample["output_tokens"], *final_flip
    )
    transformed["action_ids"] = action_ids
    transformed["actions"] = [ACTION_NAMES[action_id] for action_id in action_ids]
    transformed["world_action_ids"] = world_action_ids
    transformed["world_actions"] = [
        ACTION_NAMES[action_id] for action_id in world_action_ids
    ]
    maze_flip_meta(transformed["meta"])
    return transformed


def update_statistics(
    stats: Dict[str, Counter], sample: Mapping[str, Any], split: str
) -> None:
    stats["level"][str(int(sample["meta"]["level"]))] += 1
    (_height, width, start_state) = maze_dimensions(sample["meta"])
    if split == "train":
        current_state = int(sample["input_state"])
        next_state = int(sample["output_state"])
        stats["view_transform"][
            view_transform_name(*view_flip_for_state(start_state, current_state, width))
        ] += 1
        stats["next_view_transform"][
            view_transform_name(*view_flip_for_state(start_state, next_state, width))
        ] += 1
        stats["world_action"][str(sample["world_action"])] += 1
        stats["view_action"][str(sample["action"])] += 1
        if int(sample["action_id"]) != int(sample["world_action_id"]):
            stats["changed_action_label"]["yes"] += 1
        else:
            stats["changed_action_label"]["no"] += 1
    else:
        for state in _normalize_route(sample["input_state"]):
            stats["view_transform"][
                view_transform_name(*view_flip_for_state(start_state, state, width))
            ] += 1
        for action in sample["world_actions"]:
            stats["world_action"][str(action)] += 1
        for action in sample["actions"]:
            stats["view_action"][str(action)] += 1
        for world_action, view_action in zip(
            sample["world_action_ids"], sample["action_ids"]
        ):
            key = "yes" if int(world_action) != int(view_action) else "no"
            stats["changed_action_label"][key] += 1


def new_statistics() -> Dict[str, Counter]:
    return {
        "level": Counter(),
        "view_transform": Counter(),
        "next_view_transform": Counter(),
        "world_action": Counter(),
        "view_action": Counter(),
        "changed_action_label": Counter(),
    }


def transform_jsonl(
    source: Path,
    destination: Path,
    split: str,
    max_samples: int | None,
    overwrite: bool,
) -> Dict[str, Any]:
    if destination.exists() and (not overwrite):
        with destination.open(encoding="utf-8") as handle:
            count = sum((bool(line.strip()) for line in handle))
        return {
            "split": split,
            "source": str(source),
            "destination": str(destination),
            "samples": count,
            "reused": True,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    transform = (
        transform_random_sample if split == "train" else transform_validation_sample
    )
    stats = new_statistics()
    sample_count = 0
    with source.open(encoding="utf-8") as reader, destination.open(
        "w", encoding="utf-8"
    ) as writer:
        for line in reader:
            if max_samples is not None and sample_count >= max_samples:
                break
            if not line.strip():
                continue
            transformed = transform(json.loads(line))
            writer.write(
                json.dumps(transformed, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            update_statistics(stats, transformed, split)
            sample_count += 1
    return {
        "split": split,
        "source": str(source),
        "destination": str(destination),
        "samples": sample_count,
        "statistics": {
            name: dict(sorted(counter.items()))
            for (name, counter) in stats.items()
            if counter
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_source",
        default="dataset/maze/tokenized_dataset/SFT_random/train_dataset.jsonl",
    )
    parser.add_argument(
        "--validation_source",
        default="dataset/maze/tokenized_dataset/SFT/test_dataset.jsonl",
    )
    parser.add_argument("--output_root", default="dataset/maze_flip/tokenized_dataset")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_validation_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    output_root = Path(args.output_root)
    log_stage("1/3 Prepare the MazeFlip training dataset")
    train = transform_jsonl(
        Path(args.train_source),
        output_root / "SFT_random/train_dataset.jsonl",
        "train",
        args.max_train_samples,
        args.overwrite,
    )
    log_stage("2/3 Prepare the MazeFlip validation dataset")
    validation = transform_jsonl(
        Path(args.validation_source),
        output_root / "SFT/test_dataset.jsonl",
        "validation",
        args.max_validation_samples,
        args.overwrite,
    )
    manifest = {
        "dataset": "maze_flip",
        "version": DATASET_VERSION,
        "token_grid_size": DEFAULT_TOKEN_GRID_SIZE,
        "rule": {
            "LEFT_or_RIGHT": "toggle top-bottom image flip",
            "UP_or_DOWN": "toggle left-right image flip",
            "world_route": "unchanged",
            "action_labels": "remapped into the current view coordinate frame",
        },
        "splits": {"train": train, "validation": validation},
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_stage("3/3 Dataset preparation completed")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
