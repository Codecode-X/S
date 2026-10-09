import argparse
import csv
import datetime as dt
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
VQA_GENERATION_SCRIPT = "tools/vqa/generate_vqav2.py"
VQA_JUDGE_SCRIPT = "tools/vqa/judge_answers.py"
MAZE_MCQ_SCRIPT = "tools/vqa/evaluate_maze_mcq.py"
VARIANTS = {
    "ssvr": dict(
        alpha=0.7, state="recurrent_implicit_state", head="text_token", tables=[6, 9]
    ),
    "alpha_1p0": dict(
        alpha=1.0, state="recurrent_implicit_state", head="text_token", tables=[6]
    ),
    "alpha_0p4": dict(
        alpha=0.4, state="recurrent_implicit_state", head="text_token", tables=[6]
    ),
    "ssvr_head": dict(
        alpha=0.7, state="recurrent_implicit_state", head="linear", tables=[9]
    ),
}
EXPERIMENT_KINDS = {
    "suite",
    "ablation",
    "image_scale",
    "invert",
    "flip_vertical",
    "flip_horizontal",
    "flip_both",
    "paraphrase",
    "native_maze_vqa",
    "vqa",
    "attention_maze",
    "attention_frozenlake",
    "attention_minibehaviour",
}
PARAPHRASE = "Find a path through the maze and reach the goal without going through walls. The available next-action choices are ordered as UP, DOWN, LEFT, RIGHT. Pick one next action only."
LIMITATIONS = [
    "Training, planning evaluation and VQA/MCQ tools use the current project source.",
    "Original training: model initialization seed 2026; Trainer seed/data_seed retain original defaults (42/None); cosine for 10 epochs, stop after 5; LR 1.5e-4; warmup 0.1; LoRA 32/64/0.1.",
    "GRU encodes initial context plus action history; coordinates, state_step and explicit flags are not used. All four variants retain GRU. No step ablation is run.",
    "VQAv2: original JSON order, random 1000 indices with seed 2026, image thumbnail max 256, greedy64, Answer directly.; judge qwen3.6-flash via DashScope.",
    "Baseline/head judge compares native and trained answers; alpha variants use the original model-only judge protocol.",
    "VQAv2 accuracy is LLM-judge semantic accuracy in native generation mode, not the official VQA consensus score.",
    "Four Maze ablation models plus FrozenLake and MiniBehaviour alpha=0.7 GRU models for attention. Paper state/step ablation deliberately excluded.",
    "These are new GRU-model experiments, not measured reproductions of explicit-state paper scores. Raw results remain separate from historical results.",
]


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path, obj):
    write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def env_path(name, default=""):
    value = os.environ.get(name, default)
    return str(Path(value).expanduser().resolve()) if value else ""


def main_run_checkpoints(root):
    root = Path(root)
    config = json.loads((root / "metadata/config.json").read_text())
    return {
        task: (root / "metadata" / f"{task}_checkpoint.txt").read_text().strip()
        for task in config.get("TASKS", "maze,frozenlake,minibehaviour").split(",")
    }


def configuration():
    e = os.environ
    vqa_root = Path(
        env_path("VQA_DATA_ROOT", "/home/usr/UU/data/VQAv2_val/downloads")
    )
    model = e.get("QWEN_MODEL_PATH", "")
    if not model:
        for adapter in sorted(
            (ROOT / "models/SSVR").glob("*/backbone/adapter_config.json")
        ):
            candidate = json.loads(adapter.read_text()).get(
                "base_model_name_or_path", ""
            )
            if (Path(candidate) / "config.json").is_file():
                model = candidate
                break
    model = model or "Qwen/Qwen2.5-VL-7B-Instruct"
    data_root = Path(env_path("DATA_ROOT", str(ROOT / "dataset")))
    variants = e.get("ABLATIONS", ",".join(VARIANTS)).split(",")
    if len(set(variants)) != len(variants) or any(
        (x not in VARIANTS for x in variants)
    ):
        raise ValueError(
            f"ABLATIONS must be a unique comma-separated subset of {list(VARIANTS)}"
        )
    experiment = e.get("EXPERIMENT_KIND", "suite")
    if experiment not in EXPERIMENT_KINDS:
        raise ValueError(f"Unknown EXPERIMENT_KIND: {experiment}")
    if experiment != "suite" and len(variants) != 1:
        raise ValueError(
            "An independent experiment requires exactly one ABLATIONS variant"
        )
    if experiment in ("attention_frozenlake", "attention_minibehaviour"):
        variants = []
    scope = e.get("ABLATION_SCOPE", "full")
    if scope not in ("full", "planning"):
        raise ValueError("ABLATION_SCOPE must be full or planning")
    cfg = dict(
        schema_version=7,
        experiment=experiment,
        model_family="gru",
        project=str(ROOT),
        data_root=str(data_root),
        variants=variants,
        scope=scope,
        model=model,
        processor=e.get("QWEN_PROCESSOR_PATH", model),
        image_root=env_path(
            "QWEN_VISUAL_IMAGE_ROOT",
            str(ROOT / "output/qwen_visual_initial_maps_vqvae"),
        ),
        train=env_path(
            "TRAIN_DATASET",
            str(data_root / "maze/tokenized_dataset/SFT_random/train_dataset.jsonl"),
        ),
        test=env_path(
            "TEST_DATASET",
            str(data_root / "maze/tokenized_dataset/SFT/test_dataset.jsonl"),
        ),
        num_gpus=int(e.get("NUM_GPUS", 8)),
        batch_size=int(e.get("PER_DEVICE_BATCH_SIZE", 16)),
        grad_accum=int(e.get("GRADIENT_ACCUMULATION_STEPS", 1)),
        seed=int(e.get("SEED", 2026)),
        learning_rate=float(e.get("LEARNING_RATE", 0.00015)),
        epochs=float(e.get("NUM_EPOCHS", 10)),
        stop_epochs=float(e.get("STOP_AFTER_EPOCHS", 5)),
        max_samples=int(e.get("EVAL_MAX_SAMPLES", 1000)),
        port=int(e.get("MASTER_PORT", 29621)),
        processor_use_fast=e.get("QWEN_PROCESSOR_USE_FAST", "false") == "true",
        skip_image_prep=e.get("SKIP_IMAGE_PREP", "1") == "1",
        use_kv_cache=e.get("USE_KV_CACHE", "0") == "1",
        vqvae_dir=e.get("VQVAE_DIR", ""),
        cuda_visible_devices=e.get("CUDA_VISIBLE_DEVICES"),
        paraphrase=PARAPHRASE,
        limitations=LIMITATIONS,
        vqa_data_root=str(vqa_root),
        vqa_questions=str(vqa_root / "v2_OpenEnded_mscoco_val2014_questions.json"),
        vqa_annotations=str(vqa_root / "v2_mscoco_val2014_annotations.json"),
        vqa_images=str(vqa_root / "val2014"),
        vqa_question_ids="",
        vqa_image_template=e.get(
            "VQA_IMAGE_TEMPLATE", "COCO_val2014_{image_id:012d}.jpg"
        ),
        vqa_max_samples=int(e.get("VQA_MAX_SAMPLES", 1000)),
        vqa_sample_seed=int(e.get("VQA_SAMPLE_SEED", 2026)),
        vqa_sample_strategy=e.get("VQA_SAMPLE_STRATEGY", "random"),
        vqa_max_image_size=int(e.get("VQA_MAX_IMAGE_SIZE", 256)),
        vqa_prompt=e.get("VQA_PROMPT", "Answer directly."),
        vqa_max_new_tokens=int(e.get("VQA_MAX_NEW_TOKENS", 64)),
        judge_model=e.get("VQA_JUDGE_MODEL", "qwen3.6-flash"),
        judge_base_url=e.get(
            "VQA_JUDGE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        judge_concurrency=int(e.get("VQA_JUDGE_CONCURRENCY", 16)),
    )
    cfg["analysis"] = e.get("RUN_ANALYSIS", "1") == "1"
    cfg["reuse_main_run"] = env_path("REUSE_MAIN_RUN")
    if e.get("REUSE_MAIN_MODELS", "0") == "1" and (not cfg["reuse_main_run"]):
        for candidate in sorted(
            (ROOT / "output/reproductions").glob("*"), reverse=True
        ):
            try:
                main_run_checkpoints(candidate)
            except (OSError, ValueError, KeyError):
                continue
            cfg["reuse_main_run"] = str(candidate.resolve())
            break
        if not cfg["reuse_main_run"]:
            raise ValueError(
                "No completed GRU main run found; specify REUSE_MAIN_RUN or REUSE_MAIN_MODELS=0"
            )
    cfg["reuse_checkpoints"] = (
        main_run_checkpoints(cfg["reuse_main_run"]) if cfg["reuse_main_run"] else {}
    )
    cfg["scale_ratios"] = [
        float(x)
        for x in e.get("IMAGE_SCALES", "0.7,0.8,0.9,1.0,1.1,1.2,1.3")
        .replace(" ", ",")
        .split(",")
        if x
    ]
    cfg["frozenlake_train"] = env_path(
        "FROZENLAKE_TRAIN_DATASET",
        str(data_root / "frozenlake/tokenized_dataset/SFT_random/train_dataset.jsonl"),
    )
    cfg["frozenlake_test"] = env_path(
        "FROZENLAKE_TEST_DATASET",
        str(data_root / "frozenlake/tokenized_dataset/SFT/test_dataset.jsonl"),
    )
    for task in ("minibehaviour",):
        cfg[f"{task}_train"] = env_path(
            "MINIBEHAVIOUR_TRAIN_DATASET",
            str(data_root / task / "tokenized_dataset/SFT_random/train_dataset.jsonl"),
        )
        cfg[f"{task}_test"] = env_path(
            "MINIBEHAVIOUR_TEST_DATASET",
            str(data_root / task / "tokenized_dataset/SFT/test_dataset.jsonl"),
        )
    cfg["validation"] = env_path("EVAL_DATASET", cfg["test"])
    if e.get("PARAPHRASE_FILE"):
        cfg["paraphrase"] = Path(env_path("PARAPHRASE_FILE")).read_text().strip()
    if not cfg["paraphrase"]:
        raise ValueError("Paraphrase must not be empty")
    cfg["global_batch"] = cfg["num_gpus"] * cfg["batch_size"] * cfg["grad_accum"]
    if cfg["grad_accum"] != 1:
        raise ValueError(
            "The original UU launcher fixes gradient_accumulation_steps=1"
        )
    if e.get("USE_KV_CACHE", "0") not in ("0", "1"):
        raise ValueError("USE_KV_CACHE must be 0 or 1")
    if cfg["vqa_sample_strategy"] not in ("random", "first"):
        raise ValueError("VQA_SAMPLE_STRATEGY must be random or first")
    cfg["trainer_seed"] = 42
    return cfg


def needed_stages(name, cfg):
    experiment = cfg.get("experiment", "suite")
    if experiment == "image_scale":
        return ["scale_" + str(x).replace(".", "p") for x in cfg["scale_ratios"]]
    if experiment in (
        "invert",
        "flip_vertical",
        "flip_horizontal",
        "flip_both",
        "paraphrase",
        "native_maze_vqa",
        "vqa",
    ):
        return [experiment]
    if experiment not in ("suite", "ablation"):
        return []
    stages = ["maze"]
    if 6 in VARIANTS[name]["tables"]:
        stages += ["paraphrase", "scale_0p7", "scale_1p3"]
    if name == "ssvr" and cfg.get("analysis"):
        stages += ["scale_" + str(x).replace(".", "p") for x in cfg["scale_ratios"]]
        stages += [
            "invert",
            "flip_vertical",
            "flip_horizontal",
            "flip_both",
            "native_maze_vqa",
        ]
    if cfg["scope"] == "full":
        stages += ["vqa"]
    stages = list(dict.fromkeys(stages))
    return stages


def launch(cfg, script, args, training=False):
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(cfg["num_gpus"]),
        "--main_process_port",
        str(cfg["port"]),
    ]
    if training:
        command += ["--mixed_precision", "bf16", "--dynamo_backend", "no"]
    return command + [script] + list(args)


def train_command(cfg, directory, name):
    variant = VARIANTS[name]
    args = [
        "--model_path",
        cfg["model"],
        "--processor_path",
        cfg["processor"],
        "--image_root",
        cfg["image_root"],
        "--dataset",
        cfg["train"],
        "--eval_dataset",
        cfg["validation"],
        "--output_dir",
        str(directory / "models"),
        "--run_name",
        f"gru_{name}_seed{cfg['seed']}",
        "--action_text_labels",
        "UP,DOWN,LEFT,RIGHT",
        "--num_epochs",
        str(cfg["epochs"]),
        "--stop_after_epochs",
        str(cfg["stop_epochs"]),
        "--batch_size",
        str(cfg["batch_size"]),
        "--learning_rate",
        str(cfg["learning_rate"]),
        "--eval_max_samples",
        str(cfg["max_samples"]),
        "--seed",
        str(cfg["seed"]),
        "--optimal_bias_alpha",
        str(variant["alpha"]),
        "--action_output_mode",
        variant["head"],
        "--ablation_modules",
    ]
    if cfg["processor_use_fast"]:
        args += ["--processor_use_fast"]
    return launch(cfg, "src/ssvr/training/sft.py", args, training=True)


def eval_command(cfg, directory, checkpoint, stage):
    output = directory / "evals" / stage
    if cfg.get("analysis") and (
        stage.startswith("scale_")
        or stage
        in (
            "invert",
            "flip_vertical",
            "flip_horizontal",
            "flip_both",
        )
    ):
        args = [
            "--checkpoint",
            str(checkpoint),
            "--base_model",
            cfg["model"],
            "--processor_path",
            cfg["processor"],
            "--test_dataset",
            cfg["test"],
            "--image_root",
            cfg["image_root"],
            "--output_dir",
            str(output),
            "--max_samples",
            str(cfg["max_samples"]),
            "--seed",
            str(cfg["seed"]),
            "--transform",
            (
                stage
                if stage in ("invert", "flip_vertical", "flip_horizontal", "flip_both")
                else "none"
            ),
            "--image_scale",
            (
                stage.removeprefix("scale_").replace("p", ".")
                if stage.startswith("scale_")
                else "1.0"
            ),
        ]
        if cfg["processor_use_fast"]:
            args += ["--qwen_processor_use_fast"]
        args += ["--use_kv_cache" if cfg["use_kv_cache"] else "--no-use_kv_cache"]
        return launch(cfg, "tools/analysis/evaluate_gru_transforms.py", args)
    root = directory / "selected_checkpoint"
    root.mkdir(exist_ok=True)
    link = root / checkpoint.name
    if not link.exists():
        link.symlink_to(checkpoint, target_is_directory=True)
    args = [
        "--ckpt_root",
        str(root),
        "--base_model",
        cfg["model"],
        "--backbone_type",
        "qwen25vl",
        "--qwen_visual_input",
        "--qwen_processor_path",
        cfg["processor"],
        "--qwen_visual_image_root",
        cfg["image_root"],
        "--test_dataset",
        cfg["test"],
        "--output_dir",
        str(output),
        "--doc_path",
        str(output / "metrics.md"),
        "--max_samples",
        str(cfg["max_samples"]),
        "--torch_dtype",
        "bfloat16",
        "--task",
        "maze",
        "--qwen_visual_image_scale",
        {"scale_0p7": "0.7", "scale_1p3": "1.3"}.get(stage, "1.0"),
    ]
    if cfg["processor_use_fast"]:
        args += ["--qwen_processor_use_fast"]
    args += ["--use_kv_cache" if cfg["use_kv_cache"] else "--no-use_kv_cache"]
    if stage == "paraphrase":
        args += ["--qwen_prompt_text", cfg["paraphrase"]]
    return launch(cfg, "src/ssvr/evaluation/run.py", args)


def vqa_commands(cfg, checkpoint, stage_dir, name):
    output = stage_dir / "model_outputs.json"
    args = [
        "--data_root",
        cfg["vqa_data_root"],
        "--base_model",
        cfg["model"],
        "--trained_checkpoint",
        str(checkpoint),
        "--processor_path",
        cfg["processor"],
        "--output_json",
        str(output),
        "--max_samples",
        str(cfg["vqa_max_samples"]) if cfg["vqa_max_samples"] else "all",
        "--sample_seed",
        str(cfg["vqa_sample_seed"]),
        "--sample_strategy",
        cfg["vqa_sample_strategy"],
        "--max_new_tokens",
        str(cfg["vqa_max_new_tokens"]),
        "--max_image_size",
        str(cfg["vqa_max_image_size"]),
        "--direct_answer_prompt",
        cfg["vqa_prompt"],
    ]
    model_only = name in ("alpha_1p0", "alpha_0p4")
    if cfg["processor_use_fast"]:
        args += ["--processor_use_fast"]
    if model_only:
        args += ["--skip_native_qwen"]
    generation = launch(cfg, VQA_GENERATION_SCRIPT, args, training=True)
    judge = [
        sys.executable,
        VQA_JUDGE_SCRIPT,
        "--input_json",
        str(output),
        "--output_dir",
        str(stage_dir / "llm_judge"),
        "--judge_model",
        cfg["judge_model"],
        "--base_url",
        cfg["judge_base_url"],
        "--concurrency",
        str(cfg["judge_concurrency"]),
    ]
    if model_only:
        judge += ["--model_only"]
    return (generation, judge)


def run(command, directory, label, env):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "commands.jsonl").open("a") as f:
        f.write(json.dumps(dict(stage=label, argv=command)) + "\n")
    print(
        f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] {label}: {shlex.join(command)}",
        flush=True,
    )
    with (directory / f"{label}.log").open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if rc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise RuntimeError(
                f"{label} exited {rc}; see {directory / (label + '.log')}"
            )


def checkpoint_in(directory):
    candidates = [
        p
        for p in directory.glob("checkpoint-*")
        if p.is_dir() and p.name.split("-")[-1].isdigit()
    ]
    return (
        max(candidates, key=lambda p: int(p.name.split("-")[-1]))
        if candidates
        else None
    )


def collect_summary(out, cfg):
    rows = []
    for name in cfg["variants"]:
        result_path = out / name / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            rows.append(
                dict(variant=name, status=result["status"], **result["metrics"])
            )
    write_json(out / "results.json", rows)
    columns = [
        "variant",
        "status",
        "maze_em",
        "maze_pr",
        "entropy",
        "vqa_acc",
        "paraphrase_em",
        "scale_0p7_em",
        "scale_1p3_em",
    ]
    buf = io.StringIO()
    columns += sorted({k for row in rows for k in row} - set(columns))
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    write_text(out / "results.csv", buf.getvalue())
    lines = [
        "# GRU implicit-state ablation results",
        "",
        f"Scope: {cfg['scope']}. EM/PR/VQA are percentages; entropy is in nats; empty values indicate experiments that were not run.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    (
                        str(round(row[k], 4))
                        if isinstance(row.get(k), float)
                        else str(row.get(k, "—"))
                    )
                    for k in columns
                )
            )
            + " |"
        )
    lines += ["", "## Reproduction scope", ""] + ["- " + x for x in LIMITATIONS]
    write_text(out / "results.md", "\n".join(lines) + "\n")


def execute(cfg, out, resume):
    env = os.environ.copy()
    env.update(
        QWEN_MODEL_PATH=cfg["model"],
        QWEN_PROCESSOR_PATH=cfg["processor"],
        QWEN_VISUAL_IMAGE_ROOT=cfg["image_root"],
        TRAIN_DATASET=cfg["train"],
        EVAL_DATASET=cfg["validation"],
        TEST_DATASET=cfg["test"],
        TASKS="maze",
        NUM_GPUS=str(cfg["num_gpus"]),
        PER_DEVICE_BATCH_SIZE=str(cfg["batch_size"]),
        GRADIENT_ACCUMULATION_STEPS=str(cfg["grad_accum"]),
        NUM_EPOCHS=str(cfg["epochs"]),
        STOP_AFTER_EPOCHS=str(cfg["stop_epochs"]),
        SEED=str(cfg["seed"]),
        EVAL_MAX_SAMPLES=str(cfg["max_samples"]),
        MASTER_PORT=str(cfg["port"]),
        SKIP_MERGED_EXPORT="1",
        QWEN_PROCESSOR_USE_FAST=str(cfg["processor_use_fast"]).lower(),
        SWANLAB_MODE="disabled",
        PYTHONUNBUFFERED="1",
    )
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.pop("PYTHONPATH", None)
    env.update(
        PYTHON_BIN=sys.executable,
        MODEL_PATH=cfg["model"],
        BACKBONE_TYPE="qwen25vl",
        QWEN_VISUAL_INPUT="true",
        QWEN_VISUAL_IMAGE_SIZE="256",
        DATASET_PTH=cfg["train"],
        EVAL_DATASET_PTH=cfg["validation"],
        ENABLE_STAGE1_EVAL="1",
        SFT_RANDOM_LOSS="optimal_biased_target_closer_not_closer",
        STATE_FLAG_FUSION_MODE="add",
        ACTION_TEXT_LABELS="UP,DOWN,LEFT,RIGHT",
        LEGALITY_LOSS_WEIGHT="0.0",
        PICK_DROP_GATE_LOSS_WEIGHT="0.0",
        MINIBEHAVIOUR_INTERACTION_ACTION_MASKING="false",
    )
    env.pop("SKIP_MERGED_EXPORT", None)
    for key in (
        "QWEN_PROMPT_TEXT",
        "QWEN_VISUAL_IMAGE_SCALE",
        "RESUME_FROM_CHECKPOINT",
        "OVERWRITE",
    ):
        env.pop(key, None)
    if not cfg["skip_image_prep"] and (not (out / "prepared.json").exists()):
        try:
            tasks = {"maze"} if cfg["variants"] else set()
            tasks.update(
                (
                    job["name"].removeprefix("train_")
                    for job in analysis_plan(cfg)
                    if job["name"].startswith("train_")
                )
            )
            for task in sorted(tasks):
                train = cfg["train"] if task == "maze" else cfg[f"{task}_train"]
                test = cfg["test"] if task == "maze" else cfg[f"{task}_test"]
                validation = cfg["validation"] if task == "maze" else test
                prep_env = dict(
                    env,
                    TASKS=task,
                    TRAIN_DATASET=train,
                    EVAL_DATASET=validation,
                    TEST_DATASET=test,
                )
                run(
                    ["bash", "scripts/common/prepare_images.sh"],
                    out / "logs",
                    f"prepare_{task}",
                    prep_env,
                )
            write_json(out / "prepared.json", dict(image_root=cfg["image_root"]))
        except Exception as exc:
            write_json(out / "prepare_failure.json", dict(error=str(exc)))
            print(
                f"Image preparation failed: {exc}; continuing independent experiments",
                flush=True,
            )
    env["SKIP_IMAGE_PREP"] = "1"
    for name in cfg["variants"]:
        variant = VARIANTS[name]
        directory = out / name
        directory.mkdir(exist_ok=True)
        result_path = directory / "result.json"
        if result_path.exists() and json.loads(result_path.read_text()).get(
            "status"
        ) in ("complete", "planning_only"):
            print(f"Already saved: {name}", flush=True)
            collect_summary(out, cfg)
            continue
        stage = "train"
        try:
            write_json(directory / "status.json", dict(status="running", stage=stage))
            training_done = directory / "train_complete.json"
            state_mode = "append"
            output_mode = variant["head"]
            local_env = dict(
                env,
                OUTPUT_DIR=str(directory / "models"),
                RUN_NAME=f"ablation_{name}_seed{cfg['seed']}",
                SFT_RANDOM_OPTIMAL_BIAS_ALPHA=str(variant["alpha"]),
                STATE_TOKEN_MODE=state_mode,
                ACTION_OUTPUT_MODE=output_mode,
            )
            reused = (
                cfg.get("reuse_checkpoints", {}).get("maze") if name == "ssvr" else None
            )
            if reused and (not training_done.exists()):
                write_json(
                    training_done,
                    dict(
                        checkpoint=reused, reused=True, source_run=cfg["reuse_main_run"]
                    ),
                )
                print(
                    f"Reusing existing Maze baseline: {reused}; training skipped",
                    flush=True,
                )
            if not training_done.exists():
                previous = checkpoint_in(directory / "models") if resume else None
                already_at_stop = False
                if previous:
                    previous_state = json.loads(
                        (previous / "trainer_state.json").read_text()
                    )
                    already_at_stop = (
                        float(previous_state.get("epoch") or 0) >= cfg["stop_epochs"]
                    )
                if previous and (not already_at_stop):
                    local_env["RESUME_FROM_CHECKPOINT"] = str(previous)
                if not already_at_stop:
                    run(
                        train_command(cfg, directory, name),
                        directory / "logs",
                        stage,
                        local_env,
                    )
                checkpoint = checkpoint_in(directory / "models")
                if checkpoint is None:
                    raise RuntimeError(
                        "Training returned success without a complete checkpoint"
                    )
                write_json(training_done, dict(checkpoint=str(checkpoint)))
            checkpoint = Path(json.loads(training_done.read_text())["checkpoint"])
            metrics = {}
            failures = {}
            for stage in needed_stages(name, cfg):
                write_json(
                    directory / "status.json", dict(status="running", stage=stage)
                )
                stage_dir = directory / "evals" / stage
                saved_metric = stage_dir / "saved_metrics.json"
                if saved_metric.exists():
                    metrics.update(json.loads(saved_metric.read_text()))
                    continue
                try:
                    if stage == "vqa":
                        if not os.environ.get("DASHSCOPE_API_KEY"):
                            raise ValueError("Missing DASHSCOPE_API_KEY for VQA judge")
                        (native, judge) = vqa_commands(cfg, checkpoint, stage_dir, name)
                        run(native, directory / "logs", "vqa_generate", env)
                        run(judge, directory / "logs", "vqa_judge", env)
                        judged = json.loads(
                            (stage_dir / "llm_judge/summary.json").read_text()
                        )
                        current = dict(vqa_acc=100 * float(judged["model_accuracy"]))
                        if judged.get("native_accuracy") is not None:
                            current["native_vqa_acc"] = 100 * float(
                                judged["native_accuracy"]
                            )
                            current["vqa_delta"] = (
                                current["vqa_acc"] - current["native_vqa_acc"]
                            )
                    elif stage == "native_maze_vqa":
                        command = launch(
                            cfg,
                            MAZE_MCQ_SCRIPT,
                            [
                                "--base_model",
                                cfg["model"],
                                "--trained_checkpoint",
                                str(checkpoint),
                                "--processor_path",
                                cfg["processor"],
                                "--test_dataset",
                                cfg["test"],
                                "--image_root",
                                cfg["image_root"],
                                "--max_samples",
                                str(cfg["max_samples"]),
                                "--max_new_tokens",
                                "4",
                                "--output_dir",
                                str(stage_dir),
                            ],
                        )
                        if cfg["processor_use_fast"]:
                            command += ["--processor_use_fast"]
                        run(command, directory / "logs", stage, env)
                        current = {}
                        for label in ("base", "trained"):
                            with (stage_dir / f"{label}_summary.csv").open() as handle:
                                rows = list(csv.DictReader(handle))
                            if not rows:
                                raise ValueError("Empty native Maze VQA results")
                            write_json(stage_dir / f"{label}_summary.json", rows)
                            overall = next((r for r in rows if r["group"] == "overall"))
                            for key in ("optimal_acc", "legal_acc", "parse_rate"):
                                if key in overall:
                                    current[f"native_maze_{label}_{key}"] = 100 * float(
                                        float(overall[key])
                                    )
                        current["native_maze_vqa_saved"] = True
                    else:
                        eval_env = dict(
                            env,
                            OUTPUT_DIR=str(stage_dir),
                            QWEN_VISUAL_IMAGE_SCALE="1.0",
                        )
                        if stage == "paraphrase":
                            eval_env["QWEN_PROMPT_TEXT"] = cfg["paraphrase"]
                        if stage in ("scale_0p7", "scale_1p3"):
                            eval_env["QWEN_VISUAL_IMAGE_SCALE"] = (
                                "0.7" if stage == "scale_0p7" else "1.3"
                            )
                        run(
                            eval_command(cfg, directory, checkpoint, stage),
                            directory / "logs",
                            stage,
                            eval_env,
                        )
                        rows = json.loads((stage_dir / "summary.json").read_text())
                        if isinstance(rows, dict):
                            rows = [
                                dict(rows["overall"], samples=rows["overall"]["count"])
                            ]
                        if len(rows) != 1 or rows[0].get("samples", 0) <= 0:
                            raise ValueError(f"Invalid evaluator summary: {stage_dir}")
                        row = rows[0]
                        row["path"] = str(checkpoint)
                        if not stage.startswith("scale_"):
                            write_json(stage_dir / "summary.json", rows)
                        current = {stage + "_em": 100 * float(row["em"])}
                        for metric in ("pr", "legal_rate"):
                            if metric in row:
                                current[stage + "_" + metric] = 100 * float(row[metric])
                        if stage == "maze":
                            current.update(
                                maze_pr=100 * float(row["pr"]),
                                entropy=float(row["entropy"]),
                            )
                    write_json(saved_metric, current)
                    metrics.update(current)
                    write_json(directory / "partial_results.json", metrics)
                    (stage_dir / "failure.json").unlink(missing_ok=True)
                except Exception as exc:
                    failures[stage] = str(exc)
                    write_json(
                        stage_dir / "failure.json",
                        dict(status="failed", stage=stage, error=str(exc)),
                    )
                    write_json(directory / "partial_results.json", metrics)
                    print(f"FAILED {name}/{stage}: {exc}; continuing", flush=True)
            status = "complete" if cfg["scope"] == "full" else "planning_only"
            if failures:
                status = "partial_failure"
            write_json(
                result_path,
                dict(
                    failures=failures,
                    variant=name,
                    settings=variant,
                    status=status,
                    checkpoint=str(checkpoint),
                    metrics=metrics,
                    model_family="gru",
                    completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                ),
            )
            collect_summary(out, cfg)
            write_json(directory / "status.json", dict(status=status, stage="saved"))
            print(f"Saved {name}: {metrics}\nResults: {result_path}", flush=True)
        except BaseException as exc:
            write_json(
                directory / "status.json",
                dict(status="failed", stage=stage, error=str(exc)),
            )
            collect_summary(out, cfg)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            write_json(
                result_path,
                dict(
                    variant=name,
                    settings=variant,
                    status="failed",
                    metrics={},
                    failures={stage: str(exc)},
                ),
            )
            collect_summary(out, cfg)
            print(
                f"FAILED {name}/{stage}: {exc}; continuing with next variant",
                flush=True,
            )


def analysis_plan(cfg):
    if not cfg.get("analysis"):
        return []
    jobs = [
        dict(name="train_frozenlake", section="6.1", depends=[]),
        dict(name="attention_maze", section="6.1 Figure 9", depends=["ssvr/train"]),
        dict(
            name="attention_frozenlake",
            section="6.1 Figure 9",
            depends=["train_frozenlake"],
        ),
        dict(name="train_minibehaviour", section="6.1", depends=[]),
        dict(
            name="attention_minibehaviour",
            section="6.1 Figure 9",
            depends=["train_minibehaviour"],
        ),
    ]
    experiment = cfg.get("experiment", "suite")
    selected = {
        "attention_maze": {"attention_maze"},
        "attention_frozenlake": {"train_frozenlake", "attention_frozenlake"},
        "attention_minibehaviour": {"train_minibehaviour", "attention_minibehaviour"},
    }
    if experiment != "suite":
        jobs = [job for job in jobs if job["name"] in selected.get(experiment, set())]
    for job in jobs:
        if job["name"].startswith("train_"):
            task = job["name"].removeprefix("train_")
            if cfg.get("reuse_checkpoints", {}).get(task):
                job["operation"] = "reuse_checkpoint"
                job["checkpoint"] = cfg["reuse_checkpoints"][task]
    return jobs


def execute_analysis(cfg, out, resume):
    if not cfg.get("analysis"):
        return
    root = out / "analysis"
    root.mkdir(exist_ok=True)
    env = dict(os.environ, SWANLAB_MODE="disabled", PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    for key in (
        "RESUME_FROM_CHECKPOINT",
        "QWEN_PROMPT_TEXT",
        "QWEN_VISUAL_IMAGE_SCALE",
    ):
        env.pop(key, None)

    def trained(task):
        path = (
            out / "ssvr/train_complete.json"
            if task == "maze"
            else root / f"train_{task}/train_complete.json"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Required {task} GRU training did not complete: {path}"
            )
        return Path(json.loads(path.read_text())["checkpoint"])

    rows = []
    for item in analysis_plan(cfg):
        name = item["name"]
        directory = root / name
        directory.mkdir(exist_ok=True)
        result_path = directory / "result.json"
        if (
            result_path.exists()
            and json.loads(result_path.read_text())["status"] == "complete"
        ):
            rows.append(json.loads(result_path.read_text()))
            write_json(root / "results.json", rows)
            continue
        record = dict(item, status="running")
        write_json(directory / "status.json", record)
        try:
            if name.startswith("train_"):
                task = name.removeprefix("train_")
                reused = cfg.get("reuse_checkpoints", {}).get(task)
                if reused:
                    write_json(
                        directory / "train_complete.json",
                        dict(
                            checkpoint=reused,
                            reused=True,
                            source_run=cfg["reuse_main_run"],
                        ),
                    )
                    record.update(status="complete", checkpoint=reused, reused=True)
                    write_json(result_path, record)
                    write_json(directory / "status.json", record)
                    rows.append(record)
                    write_json(root / "results.json", rows)
                    print(f"Reusing {task}: {reused}; training skipped", flush=True)
                    continue
                task_cfg = dict(
                    cfg, train=cfg[f"{task}_train"], validation=cfg[f"{task}_test"]
                )
                command = train_command(task_cfg, directory, "ssvr")
                command[command.index("--run_name") + 1] = (
                    f"gru_{task}_alpha07_seed{cfg['seed']}"
                )
                if task == "minibehaviour":
                    command[command.index("--action_text_labels") + 1] = (
                        "UP,DOWN,LEFT,RIGHT,PICK,DROP"
                    )
                previous = checkpoint_in(directory / "models") if resume else None
                train_env = dict(env)
                at_stop = False
                if previous:
                    at_stop = (
                        float(
                            json.loads(
                                (previous / "trainer_state.json").read_text()
                            ).get("epoch")
                            or 0
                        )
                        >= cfg["stop_epochs"]
                    )
                    if not at_stop:
                        train_env["RESUME_FROM_CHECKPOINT"] = str(previous)
                if not at_stop:
                    run(command, directory / "logs", "train", train_env)
                checkpoint = checkpoint_in(directory / "models")
                if checkpoint is None:
                    raise ValueError("Training did not produce a complete checkpoint")
                write_json(
                    directory / "train_complete.json", dict(checkpoint=str(checkpoint))
                )
                record["checkpoint"] = str(checkpoint)
            elif name.startswith("attention_"):
                task = name.removeprefix("attention_")
                checkpoint = trained(task)
                test = cfg["test"] if task == "maze" else cfg[f"{task}_test"]
                command = [
                    sys.executable,
                    "tools/visualization/visualize_implicit_attention.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--base_model",
                    cfg["model"],
                    "--processor_path",
                    cfg["processor"],
                    "--test_dataset",
                    test,
                    "--image_root",
                    cfg["image_root"],
                    "--output_dir",
                    str(directory / "figures"),
                    "--markdown_path",
                    str(directory / "attention.md"),
                    "--attribution_mode",
                    "grad_x_attention",
                    "--cases_per_bucket",
                    "1",
                    "--seed",
                    str(cfg["seed"]),
                ]
                if task == "minibehaviour":
                    command += [
                        "--minibehaviour_split_attribution",
                        "--min_correct_move_steps",
                        "1",
                    ]
                run(command, directory / "logs", name, env)
                if not (directory / "attention.md").is_file():
                    raise ValueError("Attention report not saved")
                record["artifact"] = str(directory / "attention.md")
            record["status"] = "complete"
        except Exception as exc:
            record.update(
                status=(
                    "skipped_dependency"
                    if isinstance(exc, FileNotFoundError) and "Required" in str(exc)
                    else "failed"
                ),
                error=str(exc),
            )
            print(f"FAILED {name}: {exc}; continuing", flush=True)
        record["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(result_path, record)
        write_json(directory / "status.json", record)
        rows.append(record)
        write_json(root / "results.json", rows)
        write_text(
            root / "results.md",
            "# Analysis jobs\n\n| Experiment | Status | Error |\n|---|---|---|\n"
            + "".join(
                (
                    f"| {r['name']} | {r['status']} | {r.get('error', '')} |\n"
                    for r in rows
                )
            ),
        )
        print(f"Saved analysis {name}: {record['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the selected paper experiment; full scope includes VQAv2 and an LLM judge."
    )
    parser.add_argument("mode", nargs="?", default="run", choices=("run",))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing RUN_DIR and skip saved stages",
    )
    args = parser.parse_args()
    cfg = configuration()
    default_run = (
        ROOT
        / "output/gru_ablations"
        / (
            dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + f"_{os.getpid()}"
        )
    )
    out = Path(env_path("RUN_DIR", str(default_run)))
    out.mkdir(parents=True, exist_ok=args.resume)
    write_json(out / "config.json", cfg)
    collect_summary(out, cfg)
    write_json(out / "status.json", dict(status="running"))
    try:
        execute(cfg, out, args.resume)
        execute_analysis(cfg, out, args.resume)
    except BaseException as exc:
        write_json(out / "status.json", dict(status="failed", error=str(exc)))
        raise
    failures = []
    for path in list(out.glob("*/result.json")) + list(
        out.glob("analysis/*/result.json")
    ):
        row = json.loads(path.read_text())
        if row["status"] not in ("complete", "planning_only"):
            failures.append(
                dict(
                    path=str(path),
                    status=row["status"],
                    errors=row.get("failures", row.get("error")),
                )
            )
    write_json(out / "failures.json", failures)
    status = (
        "completed_with_errors"
        if failures
        else "complete" if cfg["scope"] == "full" else "planning_only"
    )
    write_json(
        out / "status.json", dict(status=status, failed_experiments=len(failures))
    )
    print(
        f"Suite finished: {status}; results: {out / 'results.md'}; analysis: {out / 'analysis/results.md'}; failures: {out / 'failures.json'}"
    )


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        sys.exit(str(exc))
