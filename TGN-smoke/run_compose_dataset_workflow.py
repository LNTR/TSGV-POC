#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


SERVICE_PORTS = {
    "visual-preprocessing-service": 8001,
    "visual-feature-extraction-service": 8002,
    "text-processing-service": 8003,
    "training-service": 8004,
    "inference-service": 8005,
    "evaluation-service": 8006,
}
COMPOSE_SERVICES = [
    "mongodb",
    "redis",
    "visual-preprocessing-service",
    "visual-feature-extraction-service",
    "text-processing-service",
    "training-service",
    "inference-service",
    "evaluation-service",
]
PHASE_SERVICES = {
    "preprocess": ["visual-preprocessing-service", "visual-feature-extraction-service"],
    "text_processing": ["text-processing-service"],
    "training": ["training-service"],
    "inference": ["inference-service"],
    "evaluation": ["evaluation-service", "inference-service"],
}


def json_request(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 300) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {details}") from exc
    if not body:
        return {}
    return json.loads(body)


def wait_for_services(base_url: str, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    pending = dict(SERVICE_PORTS)
    while pending:
        for service, port in list(pending.items()):
            try:
                json_request("GET", f"{base_url}:{port}/registry/self", timeout=5)
            except Exception:
                continue
            pending.pop(service, None)
        if not pending:
            return
        if time.time() >= deadline:
            missing = ", ".join(sorted(pending))
            raise TimeoutError(f"timed out waiting for services: {missing}")
        time.sleep(1)


def shared_uri(*parts: str) -> str:
    cleaned = "/".join(part.strip("/") for part in parts if part)
    return f"shared://{cleaned}"


def service_container_name(service_name: str) -> str:
    return f"tgn-implementation-{service_name}-1"


def parse_size_to_bytes(raw_value: str) -> int:
    value = raw_value.strip().replace(" ", "")
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)([A-Za-z]+)?", value)
    if match is None:
        return 0

    number = float(match.group(1))
    unit = (match.group(2) or "B").lower()
    factors = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }
    return int(number * factors.get(unit, 1))


def format_bytes(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"

    value = float(num_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    return f"{value:.2f} {units[unit_index]}"


def collect_memory_snapshot(services: list[str]) -> dict[str, dict[str, Any]]:
    if not services:
        return {}

    container_names = [service_container_name(service) for service in services]
    command = [
        "docker",
        "stats",
        "--no-stream",
        "--format",
        "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}",
        *container_names,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return {}

    name_to_service = {service_container_name(service): service for service in services}
    snapshot: dict[str, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        container_name, cpu_percent, memory_usage, memory_percent = parts
        service_name = name_to_service.get(container_name)
        if service_name is None:
            continue

        used_raw, _, limit_raw = memory_usage.partition("/")
        cpu_value = cpu_percent.strip().removesuffix("%")
        percent_value = memory_percent.strip().removesuffix("%")
        snapshot[service_name] = {
            "container_name": container_name,
            "cpu_percent": float(cpu_value) if cpu_value else 0.0,
            "memory_usage_bytes": parse_size_to_bytes(used_raw),
            "memory_limit_bytes": parse_size_to_bytes(limit_raw),
            "memory_percent": float(percent_value) if percent_value else 0.0,
        }
    return snapshot


def collect_gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {
            "available": False,
            "gpu_count": 0,
            "memory_used_mb": 0,
            "memory_total_mb": 0,
            "utilization_percent_max": 0.0,
            "per_gpu": [],
        }

    if result.returncode != 0:
        return {
            "available": False,
            "gpu_count": 0,
            "memory_used_mb": 0,
            "memory_total_mb": 0,
            "utilization_percent_max": 0.0,
            "per_gpu": [],
        }

    per_gpu: list[dict[str, Any]] = []
    total_used_mb = 0
    total_memory_mb = 0
    max_utilization = 0.0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        gpu_index, used_mb, total_mb, util_percent = parts
        try:
            used_value = int(float(used_mb))
            total_value = int(float(total_mb))
            util_value = float(util_percent)
        except ValueError:
            continue

        per_gpu.append(
            {
                "gpu_index": gpu_index,
                "memory_used_mb": used_value,
                "memory_total_mb": total_value,
                "utilization_percent": util_value,
            }
        )
        total_used_mb += used_value
        total_memory_mb += total_value
        max_utilization = max(max_utilization, util_value)

    return {
        "available": bool(per_gpu),
        "gpu_count": len(per_gpu),
        "memory_used_mb": total_used_mb,
        "memory_total_mb": total_memory_mb,
        "utilization_percent_max": max_utilization,
        "per_gpu": per_gpu,
    }


def flatten_gpu_snapshot(per_gpu: list[dict[str, Any]]) -> str:
    if not per_gpu:
        return ""
    return ";".join(
        f"{entry['gpu_index']}:{entry['memory_used_mb']}/{entry['memory_total_mb']}MB@{entry['utilization_percent']:.1f}%"
        for entry in per_gpu
    )


def sample_phase_memory(
    services: list[str],
    interval_sec: float,
    stop_event: threading.Event,
    samples: list[dict[str, dict[str, Any]]],
    timeline_rows: list[dict[str, Any]],
    workflow_started_at: float,
    phase_name: str,
) -> None:
    while not stop_event.is_set():
        snapshot = collect_memory_snapshot(services)
        if snapshot:
            samples.append(snapshot)
            gpu_snapshot = collect_gpu_snapshot()
            captured_at = datetime.now(timezone.utc).isoformat()
            elapsed_sec = round(time.perf_counter() - workflow_started_at, 3)
            for service_name, service_sample in snapshot.items():
                timeline_rows.append(
                    {
                        "timestamp_utc": captured_at,
                        "elapsed_sec": elapsed_sec,
                        "phase": phase_name,
                        "service": service_name,
                        "container_name": service_sample["container_name"],
                        "cpu_percent": round(float(service_sample.get("cpu_percent", 0.0)), 3),
                        "memory_usage_bytes": int(service_sample.get("memory_usage_bytes", 0)),
                        "memory_limit_bytes": int(service_sample.get("memory_limit_bytes", 0)),
                        "memory_percent": round(float(service_sample.get("memory_percent", 0.0)), 3),
                        "gpu_available": gpu_snapshot["available"],
                        "gpu_count": int(gpu_snapshot["gpu_count"]),
                        "gpu_memory_used_mb": int(gpu_snapshot["memory_used_mb"]),
                        "gpu_memory_total_mb": int(gpu_snapshot["memory_total_mb"]),
                        "gpu_utilization_percent_max": round(float(gpu_snapshot["utilization_percent_max"]), 3),
                        "gpu_per_device": flatten_gpu_snapshot(gpu_snapshot["per_gpu"]),
                    }
                )
        stop_event.wait(interval_sec)


def record_phase_stats(
    phase_stats: dict[str, dict[str, Any]],
    phase_name: str,
    duration_sec: float,
    samples: list[dict[str, dict[str, Any]]],
) -> None:
    phase_entry = phase_stats.setdefault(
        phase_name,
        {
            "duration_sec": 0.0,
            "calls": 0,
            "services": {},
        },
    )
    phase_entry["duration_sec"] += duration_sec
    phase_entry["calls"] += 1

    for sample in samples:
        for service_name, service_sample in sample.items():
            service_entry = phase_entry["services"].setdefault(
                service_name,
                {
                    "samples": 0,
                    "sum_memory_bytes": 0,
                    "max_memory_bytes": 0,
                    "last_memory_bytes": 0,
                    "max_memory_percent": 0.0,
                    "memory_limit_bytes": 0,
                },
            )
            memory_usage_bytes = int(service_sample.get("memory_usage_bytes", 0))
            memory_limit_bytes = int(service_sample.get("memory_limit_bytes", 0))
            memory_percent = float(service_sample.get("memory_percent", 0.0))

            service_entry["samples"] += 1
            service_entry["sum_memory_bytes"] += memory_usage_bytes
            service_entry["last_memory_bytes"] = memory_usage_bytes
            service_entry["max_memory_bytes"] = max(service_entry["max_memory_bytes"], memory_usage_bytes)
            service_entry["max_memory_percent"] = max(service_entry["max_memory_percent"], memory_percent)
            service_entry["memory_limit_bytes"] = max(service_entry["memory_limit_bytes"], memory_limit_bytes)


def finalize_phase_stats(phase_stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for phase_name, phase_entry in phase_stats.items():
        finalized_services: dict[str, dict[str, Any]] = {}
        for service_name, service_entry in phase_entry["services"].items():
            samples = max(1, int(service_entry["samples"]))
            avg_memory_bytes = int(service_entry["sum_memory_bytes"] / samples)
            finalized_services[service_name] = {
                "samples": int(service_entry["samples"]),
                "avg_memory_bytes": avg_memory_bytes,
                "avg_memory_human": format_bytes(avg_memory_bytes),
                "max_memory_bytes": int(service_entry["max_memory_bytes"]),
                "max_memory_human": format_bytes(int(service_entry["max_memory_bytes"])),
                "last_memory_bytes": int(service_entry["last_memory_bytes"]),
                "last_memory_human": format_bytes(int(service_entry["last_memory_bytes"])),
                "memory_limit_bytes": int(service_entry["memory_limit_bytes"]),
                "memory_limit_human": format_bytes(int(service_entry["memory_limit_bytes"])),
                "max_memory_percent": round(float(service_entry["max_memory_percent"]), 2),
            }

        finalized[phase_name] = {
            "duration_sec": round(float(phase_entry["duration_sec"]), 3),
            "calls": int(phase_entry["calls"]),
            "services": finalized_services,
        }
    return finalized


def run_timed_phase(
    phase_stats: dict[str, dict[str, Any]],
    timeline_rows: list[dict[str, Any]],
    workflow_started_at: float,
    phase_name: str,
    services: list[str],
    stats_interval_sec: float,
    func,
):
    samples: list[dict[str, dict[str, Any]]] = []
    stop_event = threading.Event()
    worker = threading.Thread(
        target=sample_phase_memory,
        args=(services, stats_interval_sec, stop_event, samples, timeline_rows, workflow_started_at, phase_name),
        daemon=True,
    )
    start = time.perf_counter()
    worker.start()
    try:
        return func()
    finally:
        duration_sec = time.perf_counter() - start
        stop_event.set()
        worker.join(timeout=stats_interval_sec + 1.0)
        record_phase_stats(phase_stats, phase_name, duration_sec, samples)


def write_timeline_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "timestamp_utc",
        "elapsed_sec",
        "phase",
        "service",
        "container_name",
        "cpu_percent",
        "memory_usage_bytes",
        "memory_limit_bytes",
        "memory_percent",
        "gpu_available",
        "gpu_count",
        "gpu_memory_used_mb",
        "gpu_memory_total_mb",
        "gpu_utilization_percent_max",
        "gpu_per_device",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_compose(implementation_root: Path, args: list[str]) -> None:
    cmd = [
        "docker",
        "compose",
        "-f",
        "docker-compose.yml",
        "-f",
        "docker-compose.gpu.yml",
        *args,
    ]
    subprocess.run(cmd, cwd=implementation_root, check=True)


def build_hyperparams(max_iter: int, batch_size: int, log_every: int, valid_niter: int) -> dict[str, Any]:
    return {
        "K": 4,
        "delta": 2,
        "threshold": 0.5,
        "batch_size": batch_size,
        "lr": 0.001,
        "hidden_size_textual_lstm": 64,
        "hidden_size_visual_lstm": 64,
        "hidden_size_ilstm": 64,
        "word_embed_size": 50,
        "visual_feature_size": 4096,
        "log_every": log_every,
        "max_iter": max_iter,
        "valid_niter": valid_niter,
        "top_n_eval": 1,
        "patience": 1,
        "max_num_trial": 1,
        "lr_decay": 0.5,
        "fps": 30,
        "sample_rate": 150,
    }


def discover_video_ids(dataset_root: Path) -> list[str]:
    texts_dir = dataset_root / "texts"
    videos_dir = dataset_root / "videos"
    text_ids = {path.name.removesuffix(".aligned.tsv") for path in texts_dir.glob("*.aligned.tsv")}
    video_ids = {path.stem for path in videos_dir.glob("*.avi")}
    common = sorted(text_ids & video_ids)
    if len(common) < 3:
        raise RuntimeError(
            f"need at least 3 matching dataset items with both text and video, found {len(common)} under {dataset_root}"
        )
    return common


def split_video_ids(video_ids: list[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    shuffled = list(video_ids)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * 0.6)
    val_count = int(total * 0.2)
    test_count = total - train_count - val_count

    if train_count == 0 or val_count == 0 or test_count == 0:
        raise RuntimeError(
            f"dataset is too small for a 60/20/20 split: total={total}, train={train_count}, val={val_count}, test={test_count}"
        )

    train_ids = shuffled[:train_count]
    val_ids = shuffled[train_count : train_count + val_count]
    test_ids = shuffled[train_count + val_count :]
    return train_ids, val_ids, test_ids


def candidate_shared_paths(storage_root: Path, video_ids: list[str]) -> list[Path]:
    candidates: list[Path] = []
    for video_id in video_ids:
        candidates.extend(
            [
                storage_root / "frames" / "processed" / f"{video_id}.vp.npy",
                storage_root / "frames" / "metadata" / f"{video_id}.vp.meta.json",
                storage_root / "features" / "visual" / f"{video_id}.vf.pt",
                storage_root / "features" / "metadata" / f"{video_id}.vf.meta.json",
            ]
        )
    return candidates


def cleanup_shared_paths(paths: list[Path], preexisting: dict[Path, bool]) -> None:
    for path in paths:
        if preexisting.get(path, False):
            continue
        if path.exists():
            path.unlink()


def remove_paths(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def clear_redis_patterns(patterns: list[str]) -> None:
    if not patterns:
        return

    redis_container = service_container_name("redis")
    for pattern in patterns:
        scan_result = subprocess.run(
            ["docker", "exec", redis_container, "redis-cli", "--scan", "--pattern", pattern],
            capture_output=True,
            text=True,
        )
        if scan_result.returncode != 0:
            continue

        keys = [line.strip() for line in scan_result.stdout.splitlines() if line.strip()]
        if not keys:
            continue

        chunk_size = 256
        for start in range(0, len(keys), chunk_size):
            chunk = keys[start : start + chunk_size]
            subprocess.run(
                ["docker", "exec", redis_container, "redis-cli", "DEL", *chunk],
                check=True,
                capture_output=True,
                text=True,
            )


def ensure_feature_extraction_used_gpu(
    preprocess_response: dict[str, Any],
    storage_root: Path,
    video_id: str,
) -> str:
    forwarded = preprocess_response.get("forwarded", {})
    forwarded_metadata = forwarded.get("metadata", {}) if isinstance(forwarded, dict) else {}

    device_used = forwarded_metadata.get("device_used")
    cache_source = forwarded_metadata.get("cache_source")

    metadata_path = storage_root / "features" / "metadata" / f"{video_id}.vf.meta.json"
    if metadata_path.exists():
        persisted_metadata = json.loads(metadata_path.read_text())
        device_used = device_used or persisted_metadata.get("device_used")
        cache_source = cache_source or persisted_metadata.get("cache_source")

    if not isinstance(device_used, str) or not device_used.startswith("cuda"):
        raise RuntimeError(
            f"visual feature extraction for {video_id} did not use GPU; "
            f"device_used={device_used!r}, cache_source={cache_source!r}"
        )

    return device_used


def cleanup_text_processed_uris(storage_root: Path, uris: list[str]) -> None:
    prefix = "shared://"
    for uri in uris:
        if not uri.startswith(prefix):
            continue
        relative = uri[len(prefix) :].lstrip("/")
        path = storage_root / relative
        if path.exists():
            path.unlink()


def copy_service_file_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a deterministic 60/20/20 dataset split and run the full TGN Docker Compose workflow with GPU services."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1", help="Base host for published Compose ports.")
    parser.add_argument(
        "--feature-service-url",
        default="http://visual-feature-extraction-service:8002",
        help="Internal URL that the preprocessing container should use to reach the feature service.",
    )
    parser.add_argument("--tag", default="compose_dataset_60_20_20", help="Run folder name under TGN-smoke/.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed used to shuffle video ids.")
    parser.add_argument(
        "--video-limit",
        type=int,
        default=None,
        help="Optional cap on the number of matching dataset videos to include before splitting.",
    )
    parser.add_argument("--output-frame-size", type=int, default=224, help="Frame size sent to preprocessing.")
    parser.add_argument("--sample-every-sec", type=float, default=5.0, help="Temporal sampling rate for preprocessing.")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS used when converting aligned rows to seconds.")
    parser.add_argument("--max-iter", type=int, default=20, help="Training iterations for the workflow run.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--log-every", type=int, default=5, help="Training logging interval.")
    parser.add_argument("--valid-niter", type=int, default=5, help="Validation interval in training iterations.")
    parser.add_argument("--top-n", type=int, default=3, help="Number of segments to request during sample inference.")
    parser.add_argument("--wait-timeout", type=int, default=120, help="Seconds to wait for services to respond.")
    parser.add_argument(
        "--stats-interval-sec",
        type=float,
        default=2.0,
        help="Sampling interval in seconds for container memory statistics.",
    )
    parser.add_argument(
        "--require-feature-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require visual feature extraction to report a CUDA device; fail the workflow otherwise.",
    )
    parser.add_argument(
        "--force-feature-recompute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete cached frame/feature artifacts and clear Redis visual cache keys before preprocessing so extraction runs fresh.",
    )
    parser.add_argument("--cleanup", action="store_true", help="Delete generated workflow artifacts before exiting.")
    parser.add_argument(
        "--compose-managed",
        action="store_true",
        help="Start the TGN GPU Docker Compose services before the workflow and stop them afterward.",
    )
    parser.add_argument(
        "--compose-no-build",
        action="store_true",
        help="When used with --compose-managed, skip rebuilding and start from whatever images already exist locally.",
    )
    parser.add_argument(
        "--keep-compose-up",
        action="store_true",
        help="When used with --compose-managed, leave the Compose services running after the workflow.",
    )
    args = parser.parse_args()

    smoke_root = Path(__file__).resolve().parent
    repo_root = smoke_root.parent
    implementation_root = repo_root / "TGN" / "implementation"
    dataset_root = repo_root / "dataset"
    storage_root = implementation_root / "storage"
    run_root = smoke_root / args.tag
    service_split_root = storage_root / "splits"
    train_split_path = service_split_root / "train" / f"{args.tag}.json"
    val_split_path = service_split_root / "val" / f"{args.tag}.json"
    test_split_path = service_split_root / "test" / f"{args.tag}.json"
    model_path = storage_root / "models" / f"{args.tag}.bin"
    compose_started = False

    all_video_ids = discover_video_ids(dataset_root)
    if args.video_limit is not None:
        if args.video_limit < 3:
            raise RuntimeError("--video-limit must be at least 3")
        all_video_ids = all_video_ids[: args.video_limit]
    train_ids, val_ids, test_ids = split_video_ids(all_video_ids, seed=args.seed)
    selected_video_ids = train_ids + val_ids + test_ids

    preexisting = {path: path.exists() for path in candidate_shared_paths(storage_root, selected_video_ids)}
    run_root.mkdir(parents=True, exist_ok=True)

    generated_text_processed_uris: list[str] = []
    phase_stats: dict[str, dict[str, Any]] = {}
    timeline_rows: list[dict[str, Any]] = []
    workflow_started_at = time.perf_counter()

    try:
        if args.compose_managed:
            compose_args = ["up", "-d", *COMPOSE_SERVICES]
            if args.compose_no_build:
                compose_args.insert(2, "--no-build")
            else:
                compose_args.insert(2, "--build")
            run_compose(implementation_root, compose_args)
            compose_started = True

        wait_for_services(args.base_url, timeout_sec=args.wait_timeout)

        if args.force_feature_recompute:
            remove_paths(candidate_shared_paths(storage_root, selected_video_ids))
            clear_redis_patterns(["visual:preprocess:v1:*", "visual:features:v1:*"])

        split_records: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        evaluation_records: list[dict[str, Any]] = []
        feature_devices: dict[str, str] = {}
        sample_inference_record: dict[str, Any] | None = None

        for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            for video_id in split_ids:
                preprocess_response = run_timed_phase(
                    phase_stats,
                    timeline_rows,
                    workflow_started_at,
                    "preprocess",
                    PHASE_SERVICES["preprocess"],
                    args.stats_interval_sec,
                    lambda video_id=video_id: json_request(
                        "POST",
                        f"{args.base_url}:{SERVICE_PORTS['visual-preprocessing-service']}/jobs/preprocess",
                        payload={
                            "input_video_uri": f"file:///app/dataset/videos/{video_id}.avi",
                            "output_frame_size": args.output_frame_size,
                            "sample_every_sec": args.sample_every_sec,
                            "feature_service_url": args.feature_service_url,
                        },
                        timeout=1800,
                    ),
                )

                if args.require_feature_gpu:
                    feature_devices[video_id] = ensure_feature_extraction_used_gpu(
                        preprocess_response=preprocess_response,
                        storage_root=storage_root,
                        video_id=video_id,
                    )

                processed = run_timed_phase(
                    phase_stats,
                    timeline_rows,
                    workflow_started_at,
                    "text_processing",
                    PHASE_SERVICES["text_processing"],
                    args.stats_interval_sec,
                    lambda video_id=video_id: json_request(
                        "POST",
                        f"{args.base_url}:{SERVICE_PORTS['text-processing-service']}/jobs/process-aligned-text",
                        payload={
                            "input_alignment_uri": f"file:///app/dataset/texts/{video_id}.aligned.tsv",
                            "artifact_uri": "shared://artifacts/text/v1",
                            "video_features_uri": f"shared://features/visual/{video_id}.vf.pt",
                            "base_name_prefix": f"{args.tag}-{video_id}",
                            "fps": args.fps,
                        },
                        timeout=600,
                    ),
                )

                for record in processed.get("records", []):
                    split_records[split_name].append(
                        {
                            "base_name": record["base_name"],
                            "video_features_uri": record["video_features_uri"],
                            "text_processed_uri": record["text_processed_uri"],
                        }
                    )
                    generated_text_processed_uris.append(record["text_processed_uri"])

                    if split_name == "test":
                        evaluation_record = dict(record)
                        evaluation_record["video_id"] = video_id
                        evaluation_records.append(evaluation_record)
                        if sample_inference_record is None:
                            sample_inference_record = evaluation_record

                metadata_path = storage_root / "features" / "metadata" / f"{video_id}.vf.meta.json"
                if video_id not in feature_devices and metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text())
                    feature_devices[video_id] = metadata.get("device_used", "unknown")

        for split_name in ["train", "val", "test"]:
            if not split_records[split_name]:
                raise RuntimeError(f"{split_name} split produced no records; adjust the dataset selection and try again")

        train_split_path.parent.mkdir(parents=True, exist_ok=True)
        val_split_path.parent.mkdir(parents=True, exist_ok=True)
        test_split_path.parent.mkdir(parents=True, exist_ok=True)

        train_split_text = json.dumps(split_records["train"], indent=2) + "\n"
        val_split_text = json.dumps(split_records["val"], indent=2) + "\n"
        test_split_text = json.dumps(evaluation_records, indent=2) + "\n"
        train_split_path.write_text(train_split_text)
        val_split_path.write_text(val_split_text)
        test_split_path.write_text(test_split_text)

        training_response = run_timed_phase(
            phase_stats,
            timeline_rows,
            workflow_started_at,
            "training",
            PHASE_SERVICES["training"],
            args.stats_interval_sec,
            lambda: json_request(
                "POST",
                f"{args.base_url}:{SERVICE_PORTS['training-service']}/jobs/train",
                payload={
                    "train_split_uri": shared_uri("splits", "train", f"{args.tag}.json"),
                    "val_split_uri": shared_uri("splits", "val", f"{args.tag}.json"),
                    "features_root_uri": "shared://features/visual",
                    "output_model_uri": shared_uri("models", f"{args.tag}.bin"),
                    "hyperparams": build_hyperparams(
                        max_iter=args.max_iter,
                        batch_size=args.batch_size,
                        log_every=args.log_every,
                        valid_niter=args.valid_niter,
                    ),
                },
                timeout=3600,
            ),
        )

        sample_inference_response: dict[str, Any] | None = None
        if sample_inference_record is not None:
            sample_inference_response = run_timed_phase(
                phase_stats,
                timeline_rows,
                workflow_started_at,
                "inference",
                PHASE_SERVICES["inference"],
                args.stats_interval_sec,
                lambda: json_request(
                    "POST",
                    f"{args.base_url}:{SERVICE_PORTS['inference-service']}/infer/ground",
                    payload={
                        "model_uri": shared_uri("models", f"{args.tag}.bin"),
                        "video_features_uri": sample_inference_record["video_features_uri"],
                        "text_processed_uri": sample_inference_record["text_processed_uri"],
                        "top_n": args.top_n,
                    },
                    timeout=1200,
                ),
            )

        evaluation_response = run_timed_phase(
            phase_stats,
            timeline_rows,
            workflow_started_at,
            "evaluation",
            PHASE_SERVICES["evaluation"],
            args.stats_interval_sec,
            lambda: json_request(
                "POST",
                f"{args.base_url}:{SERVICE_PORTS['evaluation-service']}/jobs/evaluate",
                payload={
                    "job": {
                        "job_id": args.tag,
                        "trace_id": f"{args.tag}-trace",
                        "dataset": "dataset-60-20-20",
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "version": "v1",
                    },
                    "model_uri": shared_uri("models", f"{args.tag}.bin"),
                    "test_split_uri": shared_uri("splits", "test", f"{args.tag}.json"),
                    "features_root_uri": "shared://features/visual",
                    "metrics": ["R@1_IOU0.5", f"R@{args.top_n}_IOU0.5"],
                },
                timeout=3600,
            ),
        )

        host_train_split = run_root / "train.json"
        host_val_split = run_root / "val.json"
        host_test_split = run_root / "test.json"
        host_summary_path = run_root / "summary.json"
        host_timeline_csv_path = run_root / "timeline.csv"
        host_train_split.write_text(train_split_text)
        host_val_split.write_text(val_split_text)
        host_test_split.write_text(test_split_text)
        write_timeline_csv(host_timeline_csv_path, timeline_rows)

        copy_service_file_if_exists(model_path, run_root / "model.bin")
        copy_service_file_if_exists(Path(str(model_path) + ".optim"), run_root / "model.bin.optim")
        copy_service_file_if_exists(Path(str(model_path) + ".metrics.json"), run_root / "model.bin.metrics.json")

        summary = {
            "tag": args.tag,
            "dataset_root": str(dataset_root),
            "workflow_duration_sec": round(time.perf_counter() - workflow_started_at, 3),
            "video_counts": {
                "all": len(all_video_ids),
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
            },
            "split_video_ids": {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            },
            "record_counts": {
                "train": len(split_records["train"]),
                "val": len(split_records["val"]),
                "test": len(evaluation_records),
            },
            "feature_devices": feature_devices,
            "feature_gpu_required": args.require_feature_gpu,
            "feature_recompute_forced": args.force_feature_recompute,
            "training_device": training_response["metadata"].get("device"),
            "iterations_completed": training_response["metadata"].get("iterations_completed"),
            "model_uri": training_response["metadata"].get("model_uri"),
            "phase_timings_sec": {
                phase_name: round(float(phase_entry["duration_sec"]), 3)
                for phase_name, phase_entry in phase_stats.items()
            },
            "phase_memory_stats": finalize_phase_stats(phase_stats),
            "sample_inference": {
                "base_name": sample_inference_record.get("base_name") if sample_inference_record else None,
                "query_text": sample_inference_record.get("query_text") if sample_inference_record else None,
                "segments": sample_inference_response.get("segments", []) if sample_inference_response else [],
            },
            "evaluation_scores": evaluation_response.get("scores", {}),
            "files": {
                "train_split": str(host_train_split),
                "val_split": str(host_val_split),
                "test_split": str(host_test_split),
                "model": str(run_root / "model.bin"),
                "optimizer": str(run_root / "model.bin.optim"),
                "metrics": str(run_root / "model.bin.metrics.json"),
                "timeline_csv": str(host_timeline_csv_path),
                "summary": str(host_summary_path),
            },
            "cleanup_requested": args.cleanup,
        }
        host_summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        if args.cleanup:
            cleanup_shared_paths(candidate_shared_paths(storage_root, selected_video_ids), preexisting)
            cleanup_text_processed_uris(storage_root, generated_text_processed_uris)
            for path in [
                train_split_path,
                val_split_path,
                test_split_path,
                model_path,
                Path(str(model_path) + ".optim"),
                Path(str(model_path) + ".metrics.json"),
            ]:
                if path.exists():
                    path.unlink()
            if run_root.exists():
                shutil.rmtree(run_root)
        if compose_started and not args.keep_compose_up:
            run_compose(implementation_root, ["down", "--remove-orphans"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
