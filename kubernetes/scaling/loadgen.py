#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def worker(
    *,
    url: str,
    body: bytes,
    headers: dict[str, str],
    deadline: float,
    warmup_deadline: float,
    timeout_sec: float,
    results: dict[str, list[float] | int],
    lock: threading.Lock,
) -> None:
    while time.monotonic() < deadline:
        request_started = time.monotonic()
        status_code = 0
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                response.read()
                status_code = int(response.getcode())
        except urllib.error.HTTPError as exc:
            status_code = int(exc.code)
            _ = exc.read()
        except Exception:
            status_code = 0

        latency_ms = (time.monotonic() - request_started) * 1000.0
        within_measurement_window = request_started >= warmup_deadline

        with lock:
            results["requests_total"] = int(results.get("requests_total", 0)) + 1
            if within_measurement_window:
                cast_latencies = results.setdefault("latencies_ms", [])
                assert isinstance(cast_latencies, list)
                cast_latencies.append(latency_ms)
                if status_code == 200:
                    results["success_count"] = int(results.get("success_count", 0)) + 1
                else:
                    results["non_2xx_count"] = int(results.get("non_2xx_count", 0)) + 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple in-cluster concurrent HTTP POST load generator.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--payload-file", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--duration-sec", type=float, required=True)
    parser.add_argument("--warmup-sec", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    args = parser.parse_args()

    payload = json.loads(Path(args.payload_file).read_text())
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    started_at = time.monotonic()
    deadline = started_at + float(args.duration_sec)
    warmup_deadline = started_at + max(0.0, float(args.warmup_sec))

    results: dict[str, list[float] | int] = {
        "requests_total": 0,
        "success_count": 0,
        "non_2xx_count": 0,
        "latencies_ms": [],
    }
    lock = threading.Lock()
    threads: list[threading.Thread] = []

    for _ in range(max(1, int(args.concurrency))):
        thread = threading.Thread(
            target=worker,
            kwargs={
                "url": args.url,
                "body": body,
                "headers": headers,
                "deadline": deadline,
                "warmup_deadline": warmup_deadline,
                "timeout_sec": float(args.timeout_sec),
                "results": results,
                "lock": lock,
            },
            daemon=True,
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    latencies = [float(value) for value in results.get("latencies_ms", []) if isinstance(value, (int, float))]
    measured_duration = max(0.001, float(args.duration_sec) - max(0.0, float(args.warmup_sec)))
    success_count = int(results.get("success_count", 0))
    non_2xx_count = int(results.get("non_2xx_count", 0))
    measured_requests = success_count + non_2xx_count

    summary = {
        "url": args.url,
        "concurrency": int(args.concurrency),
        "duration_sec": float(args.duration_sec),
        "warmup_sec": max(0.0, float(args.warmup_sec)),
        "requests_total": int(results.get("requests_total", 0)),
        "measured_requests": measured_requests,
        "success_count": success_count,
        "non_2xx_count": non_2xx_count,
        "throughput_rps": round(measured_requests / measured_duration, 3),
        "p50_latency_ms": round(percentile(latencies, 0.50), 3),
        "p95_latency_ms": round(percentile(latencies, 0.95), 3),
        "latency_sample_count": len(latencies),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
