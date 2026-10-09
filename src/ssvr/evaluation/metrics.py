from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from ssvr.reasoning import state_to_position


def coord_to_state(coord: Sequence[int], width: int) -> int:
    if len(coord) != 2:
        raise ValueError(f"coord must have length 2, got {coord}")
    return int(coord[0]) * int(width) + int(coord[1])


def is_optimal_step(
    current_state: int,
    next_state: int,
    distance_map: Mapping[str, Any],
    target_pos: int,
) -> bool:
    current_key = str(int(current_state))
    if current_key not in distance_map:
        return False

    current_dist = int(distance_map[current_key])
    if current_dist <= 0:
        return False

    if int(next_state) == int(target_pos):
        return current_dist == 1

    next_key = str(int(next_state))
    if next_key not in distance_map:
        return False

    return int(distance_map[next_key]) == current_dist - 1


def compute_em_pr(
    start_coords: Sequence[Sequence[int]],
    expected_move: int,
    distance_map: Mapping[str, Any],
    target_pos: int,
    width: int,
) -> Tuple[float, float]:
    expected_move = int(expected_move)
    width = int(width)
    if expected_move < 0:
        raise ValueError("expected_move must be non-negative")
    if expected_move == 0:
        return 1.0, 1.0

    consecutive_optimal = 0
    for step_idx in range(expected_move):
        if step_idx + 1 >= len(start_coords):
            break

        current_state = coord_to_state(start_coords[step_idx], width)
        next_state = coord_to_state(start_coords[step_idx + 1], width)
        if not is_optimal_step(current_state, next_state, distance_map, target_pos):
            break
        consecutive_optimal += 1

    progress_rate = consecutive_optimal / expected_move
    target_coord = state_to_position(int(target_pos), width)
    reached_target_at_expected_step = (
        len(start_coords) > expected_move
        and tuple(int(v) for v in start_coords[expected_move]) == target_coord
    )
    exact_match = 1.0 if consecutive_optimal == expected_move and reached_target_at_expected_step else 0.0
    return exact_match, progress_rate


def summarize_local_action_results(results: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    totals = {
        "count": 0,
        "accuracy_count": 0,
        "em_sum": 0.0,
        "pr_sum": 0.0,
    }
    per_level: Dict[str, Dict[str, Any]] = {}

    for result in results:
        level_key = str(result["level"])
        per_level.setdefault(
            level_key,
            {"count": 0, "accuracy_count": 0, "em_sum": 0.0, "pr_sum": 0.0},
        )
        bucket = per_level[level_key]

        complete = bool(result.get("complete", False))
        em = float(result["em"])
        pr = float(result["pr"])

        totals["count"] += 1
        bucket["count"] += 1
        if complete:
            totals["accuracy_count"] += 1
            bucket["accuracy_count"] += 1
        totals["em_sum"] += em
        totals["pr_sum"] += pr
        bucket["em_sum"] += em
        bucket["pr_sum"] += pr

    def finalize(bucket: Mapping[str, Any]) -> Dict[str, Any]:
        count = int(bucket["count"])
        return {
            "count": count,
            "accuracy": float(bucket["accuracy_count"]) / count if count else 0.0,
            "em": float(bucket["em_sum"]) / count if count else 0.0,
            "pr": float(bucket["pr_sum"]) / count if count else 0.0,
        }

    return {
        "overall": finalize(totals),
        "per_level": {level: finalize(bucket) for level, bucket in sorted(per_level.items())},
    }
