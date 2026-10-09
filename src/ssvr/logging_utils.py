from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional, Tuple

log = logging.getLogger(__name__)


RUN_ID_FILENAME = ".swanlab_run_id"


def resolve_swanlab_project(default_project: str, env_key: Optional[str] = None) -> str:

    if env_key:
        value = os.environ.get(env_key)
        if value:
            return value
    value = os.environ.get("SWANLAB_PROJECT")
    if value:
        return value
    prefix = os.environ.get("SWANLAB_PROJECT_PREFIX", "")
    suffix = os.environ.get("SWANLAB_PROJECT_SUFFIX", "")
    if prefix or suffix:
        return f"{prefix}{default_project}{suffix}"
    return default_project


def _sanitize_swanlab_project_fragment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned or "run"


def _is_swanlab_project_conflict(exc: BaseException) -> bool:
    message = str(exc)
    return (
        "[Conflict]" in message
        or "409" in message
        or "\u7248\u672c\u4e0d\u517c\u5bb9" in message
        or "version" in message.lower() and "incompatible" in message.lower()
    )


def make_compatible_project_name(project: str, experiment_name: Optional[str]) -> str:

    base = _sanitize_swanlab_project_fragment(project)
    if experiment_name:
        run_fragment = _sanitize_swanlab_project_fragment(experiment_name)
        suffix = run_fragment[:48]
    else:
        suffix = "compatible"
    candidate = f"{base}_compatible_{suffix}"
    return candidate[:120]


def make_swanlab_callback(**kwargs: Any):

    from swanlab.integration.transformers import SwanLabCallback

    class CompatibleSwanLabCallback(SwanLabCallback):
        def setup(self, args: Any, state: Any, model: Any = None) -> None:
            try:
                return super().setup(args, state, model)
            except RuntimeError as exc:
                if not _is_swanlab_project_conflict(exc):
                    raise

                original_project = self._init_kwargs.get("project")
                if not original_project:
                    raise
                fallback_project = os.environ.get("SWANLAB_FALLBACK_PROJECT")
                if not fallback_project:
                    fallback_project = make_compatible_project_name(
                        str(original_project),
                        self._init_kwargs.get("experiment_name"),
                    )
                if fallback_project == original_project:
                    raise

                log.warning(
                    "SwanLab project %r is incompatible with the current SDK/service; "
                    "retrying with compatible project %r.",
                    original_project,
                    fallback_project,
                )
                self._trainer_initialized = False
                self._init_kwargs["project"] = fallback_project
                return super().setup(args, state, model)

    return CompatibleSwanLabCallback(**kwargs)


def find_swanlab_resume_id(
    run_name: str,
    output_dir: Optional[str] = None,
    swanlog_dir: str = "swanlog",
) -> Tuple[Optional[str], str]:

    if output_dir:
        run_id_file = Path(output_dir) / RUN_ID_FILENAME
        if run_id_file.exists():
            try:
                saved = run_id_file.read_text(encoding="utf-8").strip()
                if saved:
                    log.info(
                        "SwanLab resume: using saved run_id=%s (from %s)",
                        saved, run_id_file,
                    )
                    return saved, "allow"
            except Exception as e:
                log.warning("SwanLab resume: failed to read %s: %s", run_id_file, e)


    swanlog_path = Path(swanlog_dir)
    if not swanlog_path.exists():
        return None, "never"


    dir_re = re.compile(r"^run-\d{8}_\d{6}-(?P<run_id>.+)$")


    yaml_block_re = re.compile(
        r"run_name:\s*\n\s+desc:\s*['\"]?\s*\n\s+sort:\s*\d+\s*\n\s+value:\s*"
        + re.escape(run_name)
        + r"(?!\w)"
    )

    matches = []
    for run_dir in swanlog_path.iterdir():
        if not run_dir.is_dir():
            continue
        m = dir_re.match(run_dir.name)
        if not m:
            continue
        run_id = m.group("run_id")

        if _run_matches(run_dir, run_id, run_name, yaml_block_re):
            matches.append((run_dir.name, run_id))

    if not matches:
        log.info("SwanLab resume: no previous run found for %r", run_name)
        return None, "never"


    latest_name, latest_id = sorted(matches)[-1]
    log.info(
        "SwanLab resume: matched run_id=%s from swanlog (%s)", latest_id, latest_name
    )
    return latest_id, "allow"


def _run_matches(
    run_dir: Path, run_id: str, run_name: str, yaml_block_re: "re.Pattern[str]"
) -> bool:

    config_file = run_dir / "files" / "config.yaml"
    if config_file.exists():
        try:
            content = config_file.read_text(encoding="utf-8", errors="ignore")
            if yaml_block_re.search(content):
                return True
        except Exception:
            pass


    swanlab_file = run_dir / f"run-{run_id}.swanlab"
    if swanlab_file.exists():
        try:
            with open(swanlab_file, "rb") as f:
                blob = f.read()
            if run_name.encode() in blob:
                return True
        except Exception:
            pass
    return False


def save_swanlab_run_id(output_dir: Optional[str], run_id: Optional[str]) -> None:
    if not output_dir or not run_id:
        return
    try:
        path = Path(output_dir) / RUN_ID_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(run_id, encoding="utf-8")
        log.info("SwanLab: saved run_id=%s to %s", run_id, path)
    except Exception as e:
        log.warning("SwanLab: failed to save run_id to %s: %s", output_dir, e)


def get_current_run_id() -> Optional[str]:
    try:
        import swanlab

        run = swanlab.get_run()
        if run is None:
            return None

        return getattr(run, "id", None)
    except Exception:
        return None


def make_run_id_saver(output_dir: Optional[str]):
    from transformers import TrainerCallback

    class _SaveRunIdCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            run_id = get_current_run_id()
            if run_id:
                save_swanlab_run_id(output_dir, run_id)

    return _SaveRunIdCallback()
