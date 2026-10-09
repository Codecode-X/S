#!/usr/bin/env python3


from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import ssl
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import aiohttp
from tqdm import tqdm


DEFAULT_JUDGE_MODEL = "qwen3.6-flash"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def sample_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('dataset', '')}::{int(row.get('sample_index', row.get('sample_id', -1)))}"


def read_json_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list or JSONL file: {path}")
    return data


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


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def build_judge_prompt(row: Mapping[str, Any], model_only: bool = False) -> str:
    question = normalize_text(row.get("question_prompt"))
    gt_answer = normalize_text(row.get("gt_answer"))
    gt_answers = row.get("gt_answers")
    if isinstance(gt_answers, list) and gt_answers:
        gt_answer = (
            f"Primary answer: {gt_answer}\n"
            f"Human answers: {json.dumps(gt_answers, ensure_ascii=False)}"
        )
    native_answer = normalize_text(row.get("native_qwen_answer"))
    model_answer = normalize_text(row.get("model_answer"))
    if model_only:
        return f"""You are a strict but fair VQA answer judge.

Judge whether the prediction is semantically equivalent to the ground-truth answer for the question.

Rules:
- Accept concise synonyms, equivalent numbers, equivalent units, and harmless wording differences.
- Reject answers that contradict the ground truth, give a different number/entity, or add a wrong final answer.
- If the prediction contains reasoning plus a final answer, judge only the final answer.
- If the prediction is empty, irrelevant, or impossible to map to the ground truth, mark it incorrect.
- Return only valid JSON. Do not wrap it in markdown.

Question:
{question}

Ground-truth answer:
{gt_answer}

Prediction (model_answer):
{model_answer}

Return exactly this JSON schema:
{{
  "model_correct": true or false,
  "model_reason": "short reason"
}}"""
    return f"""You are a strict but fair VQA answer judge.

Judge whether each prediction is semantically equivalent to the ground-truth answer for the question.

Rules:
- Accept concise synonyms, equivalent numbers, equivalent units, and harmless wording differences.
- Reject answers that contradict the ground truth, give a different number/entity, or add a wrong final answer.
- If the prediction contains reasoning plus a final answer, judge only the final answer.
- If the prediction is empty, irrelevant, or impossible to map to the ground truth, mark it incorrect.
- Return only valid JSON. Do not wrap it in markdown.

Question:
{question}

Ground-truth answer:
{gt_answer}

Prediction A (native_qwen_answer):
{native_answer}

Prediction B (model_answer):
{model_answer}

Return exactly this JSON schema:
{{
  "native_correct": true or false,
  "model_correct": true or false,
  "native_reason": "short reason",
  "model_reason": "short reason"
}}"""


def extract_json_object(text: str) -> Dict[str, Any]:
    content = str(text).strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(content)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("Expected JSON object", content, 0)
        return parsed
    except json.JSONDecodeError:
        for match in re.finditer(r"\{", content):
            try:
                parsed, _ = decoder.raw_decode(content[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise


def resolve_ca_bundle(ca_bundle: Optional[str]) -> Optional[str]:
    candidates = [
        ca_bundle,
        os.getenv("SSL_CERT_FILE"),
        os.getenv("REQUESTS_CA_BUNDLE"),
    ]
    try:
        import certifi

        candidates.append(certifi.where())
    except Exception:
        pass
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


class ChatClient:
    def __init__(self, api_key: str, base_url: str, ca_bundle: Optional[str], verify_ssl: bool):
        self.api_key = api_key
        self.base_url = str(base_url).rstrip("/")
        if verify_ssl:
            resolved_ca_bundle = resolve_ca_bundle(ca_bundle)
            self._ssl_context = ssl.create_default_context(cafile=resolved_ca_bundle)
        else:
            self._ssl_context = ssl._create_unverified_context()

    @property
    def ssl_context(self) -> ssl.SSLContext:
        return self._ssl_context

    async def chat_async(self, session: aiohttp.ClientSession, prompt: str, model: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            if response.status >= 400:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body}")
            data = await response.json(content_type=None)
        return str(data["choices"][0]["message"]["content"])


async def llm_chat_async(
    client: ChatClient,
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    max_retries: int,
    retry_sleep: float,
) -> str:
    last_error: Optional[BaseException] = None
    for attempt in range(int(max_retries) + 1):
        try:
            return await client.chat_async(session=session, prompt=prompt, model=model)
        except BaseException as exc:
            last_error = exc
            if attempt >= int(max_retries):
                break
            await asyncio.sleep(float(retry_sleep) * (attempt + 1))
    raise RuntimeError(f"LLM judge failed after {max_retries + 1} attempts: {last_error}") from last_error


async def judge_row_async(
    client: ChatClient,
    session: aiohttp.ClientSession,
    row: Mapping[str, Any],
    model: str,
    max_retries: int,
    retry_sleep: float,
    model_only: bool = False,
) -> Dict[str, Any]:
    prompt = build_judge_prompt(row, model_only=model_only)
    raw = await llm_chat_async(
        client,
        session,
        prompt,
        model=model,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
    )
    parsed = extract_json_object(raw)
    native_correct = None if model_only else bool(parsed.get("native_correct", False))
    return {
        "key": sample_key(row),
        "dataset": row.get("dataset"),
        "sample_id": row.get("sample_id"),
        "sample_index": row.get("sample_index"),
        "question_prompt": row.get("question_prompt"),
        "image_path": row.get("image_path"),
        "gt_answer": row.get("gt_answer"),
        "gt_answers": row.get("gt_answers"),
        "native_qwen_answer": row.get("native_qwen_answer"),
        "model_answer": row.get("model_answer"),
        "native_correct": native_correct,
        "model_correct": bool(parsed.get("model_correct", False)),
        "native_reason": "" if model_only else normalize_text(parsed.get("native_reason")),
        "model_reason": normalize_text(parsed.get("model_reason")),
        "judge_model": model,
        "judge_raw": raw,
    }


def aggregate(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    total = len(rows)
    has_native = any(row.get("native_correct") is not None for row in rows)
    native_correct = sum(1 for row in rows if bool(row.get("native_correct"))) if has_native else None
    model_correct = sum(1 for row in rows if bool(row.get("model_correct")))
    by_dataset: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        dataset = str(row.get("dataset"))
        item = by_dataset.setdefault(
            dataset,
            {"count": 0, "native_correct": 0, "model_correct": 0},
        )
        item["count"] += 1
        if has_native:
            item["native_correct"] += int(bool(row.get("native_correct")))
        item["model_correct"] += int(bool(row.get("model_correct")))
    for item in by_dataset.values():
        count = max(int(item["count"]), 1)
        item["native_accuracy"] = item["native_correct"] / count if has_native else None
        item["model_accuracy"] = item["model_correct"] / count
        item["delta_model_minus_native"] = (
            item["model_accuracy"] - item["native_accuracy"] if has_native else None
        )
    return {
        "count": total,
        "native_correct": native_correct,
        "model_correct": model_correct,
        "native_accuracy": native_correct / total if total and has_native else None,
        "model_accuracy": model_correct / total if total else 0.0,
        "delta_model_minus_native": (
            (model_correct - native_correct) / total if total and has_native and native_correct is not None else None
        ),
        "model_only": not has_native,
        "by_dataset": dict(sorted(by_dataset.items())),
    }


def markdown_summary(input_json: Path, judgment_jsonl: Path, summary: Mapping[str, Any]) -> str:
    def fmt_optional(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.4f}"

    lines = [
        "# VQA LLM Judge Summary",
        "",
        f"- input: `{input_json}`",
        f"- judgments: `{judgment_jsonl}`",
        "",
        "| Split | Count | Native Acc | Model Acc | Delta |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| Overall | {} | {:.4f} | {:.4f} | {:+.4f} |".format(
            int(summary["count"]),
            float(summary["native_accuracy"] or 0.0),
            float(summary["model_accuracy"]),
            float(summary["delta_model_minus_native"] or 0.0),
        ),
    ]
    if summary.get("model_only"):
        lines[-1] = (
            f"| Overall | {int(summary['count'])} | N/A | "
            f"{float(summary['model_accuracy']):.4f} | N/A |"
        )
    for dataset, item in summary["by_dataset"].items():
        lines.append(
            f"| {dataset} | {int(item['count'])} | "
            f"{fmt_optional(item.get('native_accuracy'))} | "
            f"{float(item['model_accuracy']):.4f} | "
            f"{fmt_optional(item.get('delta_model_minus_native'))} |"
        )
    return "\n".join(lines) + "\n"


async def judge_rows_async(
    client: ChatClient,
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    model: str,
    max_retries: int,
    retry_sleep: float,
    concurrency: int,
    model_only: bool = False,
) -> None:
    semaphore = asyncio.Semaphore(max(int(concurrency), 1))
    timeout = aiohttp.ClientTimeout(total=180)
    connector = aiohttp.TCPConnector(ssl=client.ssl_context)

    async def _run_one(session: aiohttp.ClientSession, row: Mapping[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            return await judge_row_async(
                client=client,
                session=session,
                row=row,
                model=model,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
                model_only=model_only,
            )

    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        trust_env=True,
    ) as session:
        tasks = [asyncio.create_task(_run_one(session, row)) for row in rows]
        try:
            with tqdm(total=len(tasks), desc=f"Async LLM judge x{max(int(concurrency), 1)}") as bar:
                for future in asyncio.as_completed(tasks):
                    judged = await future
                    append_jsonl(output_path, judged)
                    bar.update(1)
        except BaseException:
            for task in tasks:
                task.cancel()
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api_key_env", default="DASHSCOPE_API_KEY")
    parser.add_argument(
        "--ca_bundle",
        default=None,
        help="Optional CA bundle path. Defaults to SSL_CERT_FILE/REQUESTS_CA_BUNDLE/certifi when available.",
    )
    parser.add_argument(
        "--insecure_skip_ssl_verify",
        action="store_true",
        help="Disable SSL certificate verification for environments with broken CA bundles.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", action="store_false", dest="resume")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrent LLM judge requests.",
    )
    parser.add_argument(
        "--model_only",
        action="store_true",
        help="Judge only model_answer; native_qwen_answer is optional and will not be judged.",
    )
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set the environment variable {args.api_key_env}")
    client = ChatClient(
        api_key=api_key,
        base_url=args.base_url,
        ca_bundle=args.ca_bundle,
        verify_ssl=not bool(args.insecure_skip_ssl_verify),
    )

    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "llm_judge"
    output_dir.mkdir(parents=True, exist_ok=True)
    judgment_jsonl = output_dir / "judgments.jsonl"
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"

    rows = read_json_rows(input_path)
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    missing = [
        sample_key(row)
        for row in rows
        if "model_answer" not in row
        or "gt_answer" not in row
        or (not args.model_only and "native_qwen_answer" not in row)
    ]
    if missing:
        raise ValueError(
            "Input rows must include gt_answer and model_answer"
            + ("" if args.model_only else ", plus native_qwen_answer")
            + ". "
            f"First missing keys: {missing[:5]}"
        )

    if not args.resume and judgment_jsonl.exists():
        judgment_jsonl.unlink()
    done = {row.get("key") for row in read_jsonl(judgment_jsonl)} if args.resume else set()
    pending = [row for row in rows if sample_key(row) not in done]
    print(f"Loaded {len(rows)} rows; completed={len(done)}; pending={len(pending)}; resume={args.resume}")

    asyncio.run(
        judge_rows_async(
            client=client,
            rows=pending,
            output_path=judgment_jsonl,
            model=args.judge_model,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            concurrency=args.concurrency,
            model_only=bool(args.model_only),
        )
    )

    judged_rows = read_jsonl(judgment_jsonl)
    wanted = {sample_key(row) for row in rows}
    judged_rows = [row for row in judged_rows if row.get("key") in wanted]
    summary = aggregate(judged_rows)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = markdown_summary(input_path, judgment_jsonl, summary)
    summary_md.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
