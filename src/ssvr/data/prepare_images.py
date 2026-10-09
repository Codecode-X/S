import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ssvr.reasoning import (
    SFTOptimLocalActionDataset,
    build_qwen_visual_start_tokens_index,
    is_minibehaviour_meta,
    qwen_initial_visual_state,
    qwen_visual_image_path,
    sample_group_key,
    task_state_from_position,
)


def read_jsonl(path: Path, max_samples: Optional[int] = None) -> List[Mapping[str, Any]]:
    samples: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and len(samples) >= int(max_samples):
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if not samples:
        raise ValueError(f"No samples found in {path}")
    return samples


def load_vqvae(device: str, vqvae_dir: Optional[str] = None):
    from ssvr.tokenizers.vqvae_muse import VQGANModel, get_tokenizer_muse

    if not vqvae_dir:
        return get_tokenizer_muse().to(device).eval()

    root = Path(vqvae_dir)
    config_path = root / "config.json"
    weights_path = root / "pytorch_model.bin"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"VQ-VAE directory must contain config.json and pytorch_model.bin: {root}"
        )
    with weights_path.open("rb") as handle:
        if handle.read(42).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                f"VQ-VAE weight is a Git LFS pointer instead of the real model: {weights_path}. "
                "Download the actual pytorch_model.bin before preprocessing."
            )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("_class_name", None)
    config.pop("_version", None)
    model = VQGANModel(**config)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=False))
    return model.to(device).eval()


def tensor_to_pil(pixel_values: torch.Tensor) -> Image.Image:
    image = pixel_values.detach().float().cpu().clamp(0.0, 1.0)[0]
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def decode_batch(vqvae, token_batch: Sequence[Sequence[int]], device: str) -> List[Image.Image]:
    ids = torch.tensor(token_batch, dtype=torch.long, device=device)
    with torch.inference_mode():
        decoded = vqvae.decode_code(ids)
    return [tensor_to_pil(decoded[i : i + 1]) for i in range(decoded.shape[0])]


def collect_initial_token_rows(
    jsonl_paths: Sequence[Path],
    max_samples_per_file: Optional[int],
) -> Tuple[Dict[Tuple[int, ...], Dict[str, Any]], List[Dict[str, Any]]]:
    token_rows: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    manifest_rows: List[Dict[str, Any]] = []
    for jsonl_path in jsonl_paths:
        samples = read_jsonl(jsonl_path, max_samples=max_samples_per_file)
        start_tokens_index = build_qwen_visual_start_tokens_index(samples)
        for obj in samples:
            meta = obj["meta"]
            start_state = qwen_initial_visual_state(meta, obj.get("input_state"))
            if start_state is None:
                continue
            key = (sample_group_key(meta), int(start_state))
            tokens = start_tokens_index.get(key)
            if tokens is None:
                raise ValueError(
                    f"Could not recover initial tokens for {jsonl_path}, group={key[0]}, "
                    f"start_state={start_state}"
                )
            token_key = tuple(int(token) for token in tokens)
            token_rows.setdefault(
                token_key,
                {
                    "source_files": [],
                    "level": int(meta["level"]),
                    "task": "minibehaviour"
                    if is_minibehaviour_meta(meta)
                    else "maze"
                    if isinstance(meta.get("layout"), list)
                    and meta.get("layout")
                    and isinstance(meta["layout"][0], list)
                    and meta["layout"][0]
                    and isinstance(meta["layout"][0][0], Mapping)
                    else "frozenlake",
                },
            )
            source_files = token_rows[token_key]["source_files"]
            if str(jsonl_path) not in source_files:
                source_files.append(str(jsonl_path))

        manifest_rows.append(
            {
                "source_file": str(jsonl_path),
                "sample_count": len(samples),
                "initial_image_count_in_file": len(set(start_tokens_index.values())),
            }
        )
    return token_rows, manifest_rows


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    log.info("========== [IMAGE-PREP] Initialize image preprocessing ==========")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", nargs="+", required=True, help="Input VisualPlanning JSONL files.")
    parser.add_argument("--output_root", required=True, help="Directory for hash-named PNG files.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--vqvae_dir", default=None)
    parser.add_argument("--max_samples_per_file", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if int(args.num_shards) < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= int(args.shard_index) < int(args.num_shards)):
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    log.info("========== [IMAGE-PREP] Validate input data and output directory ==========")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_paths = [Path(path) for path in args.jsonl]
    for path in jsonl_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        first_line = path.open("r", encoding="utf-8").readline()
        if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"Dataset is still a Git LFS pointer: {path}")

    log.info("========== [IMAGE-PREP] Collect and deduplicate initial visual tokens ==========")
    token_rows, source_manifest = collect_initial_token_rows(
        jsonl_paths,
        max_samples_per_file=args.max_samples_per_file,
    )
    all_tokens = sorted(token_rows)
    shard_tokens = [
        tokens
        for idx, tokens in enumerate(all_tokens)
        if idx % int(args.num_shards) == int(args.shard_index)
    ]
    pending = [
        tokens
        for tokens in shard_tokens
        if args.overwrite or not qwen_visual_image_path(output_root, tokens).exists()
    ]
    log.info(
        "========== [IMAGE-PREP] Shard %s/%s: pending %s / assigned %s ==========" ,
        args.shard_index,
        args.num_shards,
        len(pending),
        len(shard_tokens),
    )
    if pending:
        log.info("========== [IMAGE-PREP] Load VQ-VAE and start decoding ==========")
        vqvae = load_vqvae(args.device, args.vqvae_dir)
        desc = (
            "Decoding initial maps"
            if int(args.num_shards) == 1
            else f"Decoding initial maps shard {args.shard_index}/{args.num_shards}"
        )
        for start in tqdm(range(0, len(pending), int(args.batch_size)), desc=desc):
            chunk = pending[start : start + int(args.batch_size)]
            images = decode_batch(vqvae, chunk, args.device)
            for tokens, image in zip(chunk, images):
                image.save(qwen_visual_image_path(output_root, tokens))

    log.info("========== [IMAGE-PREP] Write the image cache manifest ==========")
    manifest = {
        "output_root": str(output_root),
        "source_files": source_manifest,
        "unique_initial_images": len(token_rows),
        "assigned_to_shard": len(shard_tokens),
        "decoded_this_run": len(pending),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "naming": "sha1 comma-joined initial visual tokens + .png",
    }
    manifest_name = (
        "manifest.json"
        if int(args.num_shards) == 1
        else f"manifest.shard-{int(args.shard_index):05d}-of-{int(args.num_shards):05d}.json"
    )
    (output_root / manifest_name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    log.info("========== [IMAGE-PREP] Current shard completed ==========")


if __name__ == "__main__":
    main()
