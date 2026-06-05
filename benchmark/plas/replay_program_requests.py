#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay synthetic program-level request chains against SGLang."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30003/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-programs", type=int, default=32)
    parser.add_argument("--calls-per-program", type=int, default=4)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt-length-buckets",
        default="32,128,512",
        help="Comma-separated approximate prompt word counts.",
    )
    parser.add_argument("--policy-label", default="unknown")
    parser.add_argument("--trace-id", default="plas-replay")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_prompt(program_idx: int, call_idx: int, target_words: int) -> str:
    stem = (
        f"Synthetic PLAS replay. Program {program_idx}, call {call_idx}. "
        "Answer briefly after reading the repeated context. "
    )
    filler = " ".join(
        f"context_{program_idx}_{call_idx}_{i}" for i in range(target_words)
    )
    return stem + filler


def post_json(
    url: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def usage_from_response(resp: Dict[str, Any]) -> Dict[str, int]:
    usage = resp.get("usage") if isinstance(resp, dict) else None
    if not isinstance(usage, dict):
        return {}
    out = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            out[key] = value
    return out


def run_program(
    args: argparse.Namespace, program_idx: int, buckets: List[int]
) -> Dict[str, Any]:
    program_id = f"synthetic:program-{program_idx:04d}"
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    rng = random.Random(args.seed + program_idx)
    request_rows = []
    program_start = time.perf_counter()

    for call_idx in range(args.calls_per_program):
        target_words = rng.choice(buckets)
        payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": build_prompt(program_idx, call_idx, target_words),
                }
            ],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "sglang_hints": {
                "task_id": program_id,
                "program_id": program_id,
                "trace_id": args.trace_id,
                "agent_type": "synthetic",
                "stage_id": f"call-{call_idx:02d}",
                "trace_label": args.policy_label,
            },
        }

        start = time.perf_counter()
        row: Dict[str, Any] = {
            "program_id": program_id,
            "call_idx": call_idx,
            "prompt_words": target_words,
            "ok": False,
        }
        try:
            resp = post_json(endpoint, args.api_key, payload, args.timeout_s)
            row.update(usage_from_response(resp))
            row["ok"] = True
            row["response_id"] = resp.get("id")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            row["error"] = str(exc)
        finally:
            row["latency_s"] = time.perf_counter() - start
            request_rows.append(row)

        if not row["ok"]:
            break

    program_latency_s = time.perf_counter() - program_start
    return {
        "program_id": program_id,
        "ok": all(row["ok"] for row in request_rows),
        "latency_s": program_latency_s,
        "request_count": len(request_rows),
        "requests": request_rows,
    }


def summarize(programs: List[Dict[str, Any]], elapsed_s: float) -> Dict[str, Any]:
    completed = [row for row in programs if row["ok"]]
    request_rows = [req for program in programs for req in program["requests"]]
    ok_requests = [req for req in request_rows if req["ok"]]
    latencies = sorted(row["latency_s"] for row in completed)

    def percentile(p: float) -> float | None:
        if not latencies:
            return None
        idx = min(len(latencies) - 1, max(0, int(round(p * (len(latencies) - 1)))))
        return latencies[idx]

    return {
        "elapsed_s": elapsed_s,
        "completed_programs": len(completed),
        "total_programs": len(programs),
        "ok_requests": len(ok_requests),
        "total_requests": len(request_rows),
        "programs_per_min": (
            60.0 * len(completed) / elapsed_s if elapsed_s > 0 else 0.0
        ),
        "program_latency_p50_s": percentile(0.50),
        "program_latency_p95_s": percentile(0.95),
        "program_latency_p99_s": percentile(0.99),
    }


def main() -> None:
    args = parse_args()
    buckets = [
        max(1, int(item.strip()))
        for item in args.prompt_length_buckets.split(",")
        if item.strip()
    ]
    if not buckets:
        raise SystemExit("--prompt-length-buckets must contain at least one integer")

    started = time.perf_counter()
    programs = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_concurrency)) as executor:
        futures = [
            executor.submit(run_program, args, program_idx, buckets)
            for program_idx in range(args.num_programs)
        ]
        for future in as_completed(futures):
            programs.append(future.result())

    elapsed_s = time.perf_counter() - started
    result = {
        "config": vars(args),
        "summary": summarize(programs, elapsed_s),
        "programs": sorted(programs, key=lambda row: row["program_id"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
