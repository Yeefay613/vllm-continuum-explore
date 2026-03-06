#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def build_payload(model: str, prompt: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def one_request(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        resp = session.post(url, json=payload, timeout=timeout)
        elapsed = time.perf_counter() - t0
        ok = resp.status_code == 200
        data = resp.json() if ok else {}
        usage = data.get("usage", {}) if ok else {}
        return {
            "ok": ok,
            "status_code": resp.status_code,
            "latency_s": elapsed,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "error_text": "" if ok else resp.text[:300],
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "ok": False,
            "status_code": None,
            "latency_s": elapsed,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "error_text": str(exc),
        }


def run_benchmark(args: argparse.Namespace) -> int:
    url = args.url.rstrip("/") + "/v1/chat/completions"
    payload = build_payload(args.model, args.prompt, args.max_tokens, args.temperature)

    with requests.Session() as session:
        for _ in range(args.warmup):
            _ = one_request(session, url, payload, args.timeout)

        start = time.perf_counter()
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [
                ex.submit(one_request, session, url, payload, args.timeout)
                for _ in range(args.requests)
            ]
            for fut in as_completed(futures):
                results.append(fut.result())
        total_time = time.perf_counter() - start

    latencies = [r["latency_s"] for r in results]
    successes = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]

    prompt_tokens = [r["prompt_tokens"] for r in successes if isinstance(r["prompt_tokens"], int)]
    completion_tokens = [r["completion_tokens"] for r in successes if isinstance(r["completion_tokens"], int)]
    total_tokens = [r["total_tokens"] for r in successes if isinstance(r["total_tokens"], int)]

    metrics = {
        "requests": len(results),
        "successes": len(successes),
        "failures": len(failures),
        "concurrency": args.concurrency,
        "duration_s": total_time,
        "req_per_s": (len(results) / total_time) if total_time > 0 else 0.0,
        "latency_avg_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
        "prompt_tokens_total": sum(prompt_tokens),
        "completion_tokens_total": sum(completion_tokens),
        "total_tokens_total": sum(total_tokens),
        "tokens_per_s": (sum(total_tokens) / total_time) if total_time > 0 and total_tokens else None,
    }

    report = {
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "url": args.url,
            "model": args.model,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "timeout_s": args.timeout,
            "warmup": args.warmup,
            "requests": args.requests,
            "concurrency": args.concurrency,
        },
        "metrics": metrics,
        "sample_errors": [f["error_text"] for f in failures[:3]],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"saved: {output_path}")
    print(json.dumps(metrics, indent=2))
    return 0 if len(failures) == 0 else 1


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_delta(a: float | None, b: float | None, lower_is_better: bool) -> str:
    if a is None or b is None:
        return "n/a"
    if a == 0:
        return "n/a"
    pct = ((b - a) / a) * 100.0
    sign = "-" if pct < 0 else "+"
    better = (b < a) if lower_is_better else (b > a)
    tag = "better" if better else "worse"
    return f"{sign}{abs(pct):.1f}% ({tag})"


def compare_reports(args: argparse.Namespace) -> int:
    base = load_json(args.base)
    cand = load_json(args.candidate)

    bm = base["metrics"]
    cm = cand["metrics"]

    rows = [
        ("successes", bm.get("successes"), cm.get("successes"), False),
        ("failures", bm.get("failures"), cm.get("failures"), True),
        ("req_per_s", bm.get("req_per_s"), cm.get("req_per_s"), False),
        ("latency_p50_s", bm.get("latency_p50_s"), cm.get("latency_p50_s"), True),
        ("latency_p95_s", bm.get("latency_p95_s"), cm.get("latency_p95_s"), True),
        ("latency_p99_s", bm.get("latency_p99_s"), cm.get("latency_p99_s"), True),
        ("tokens_per_s", bm.get("tokens_per_s"), cm.get("tokens_per_s"), False),
    ]

    print(f"base: {args.base} ({base.get('label', 'base')})")
    print(f"candidate: {args.candidate} ({cand.get('label', 'candidate')})")
    print("")
    print(f"{'metric':<16} {'base':>12} {'candidate':>12} {'delta':>20}")
    print("-" * 64)
    for name, a, b, lower_is_better in rows:
        if isinstance(a, float):
            a_str = f"{a:.4f}"
        else:
            a_str = str(a)
        if isinstance(b, float):
            b_str = f"{b:.4f}"
        else:
            b_str = str(b)
        print(f"{name:<16} {a_str:>12} {b_str:>12} {format_delta(a, b, lower_is_better):>20}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark and compare vLLM scheduler modes.")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a benchmark against one running server.")
    run.add_argument("--url", default="http://localhost:8100")
    run.add_argument("--model", required=True)
    run.add_argument("--prompt", default="Write a 3-sentence summary of the moon.")
    run.add_argument("--max-tokens", type=int, default=128)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--timeout", type=float, default=120.0)
    run.add_argument("--warmup", type=int, default=3)
    run.add_argument("--requests", type=int, default=50)
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--label", default="run")
    run.add_argument("--output", required=True)

    cmp_ = sub.add_parser("compare", help="Compare two benchmark JSON reports.")
    cmp_.add_argument("--base", required=True)
    cmp_.add_argument("--candidate", required=True)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "run":
        return run_benchmark(args)
    if args.cmd == "compare":
        return compare_reports(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
