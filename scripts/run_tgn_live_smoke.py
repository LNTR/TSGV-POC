#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "TGN-smoke"))

from run_compose_dataset_workflow import (  # noqa: E402
    SERVICE_PORTS,
    build_hyperparams,
    candidate_shared_paths,
    clear_redis_patterns,
    discover_video_ids,
    json_request,
    remove_paths,
    shared_uri,
    wait_for_services,
)


def choose_video_ids(dataset_root: Path, seed: int) -> tuple[str, str, str]:
    video_ids = discover_video_ids(dataset_root)
    shuffled = list(video_ids)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 3:
        raise RuntimeError("need at least 3 videos for the live smoke run")
    return shuffled[0], shuffled[1], shuffled[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small live-endpoint TGN smoke flow.")
    parser.add_argument("--base-url", default="http://127.0.0.1", help="Base host for published service ports.")
    parser.add_argument(
        "--feature-service-url",
        default="http://visual-feature-extraction-service:8002/jobs/features",
        help="Internal URL used by visual-preprocessing-service to reach feature extraction.",
    )
    parser.add_argument("--encoder", default="vgg16", help="Visual encoder to request during preprocessing.")
    parser.add_argument("--tag", default="live-smoke", help="Run tag used for model and split artifact names.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used when auto-selecting videos.")
    parser.add_argument("--train-video-id", default=None, help="Optional fixed train video id.")
    parser.add_argument("--val-video-id", default=None, help="Optional fixed val video id.")
    parser.add_argument("--test-video-id", default=None, help="Optional fixed test video id.")
    parser.add_argument("--sample-every-sec", type=float, default=3.0, help="Temporal sampling rate.")
    parser.add_argument("--output-frame-size", type=int, default=256, help="Frame size sent to preprocessing.")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS used for aligned-text conversion.")
    parser.add_argument("--wait-timeout", type=int, default=120, help="Seconds to wait for services.")
    parser.add_argument("--max-iter", type=int, default=1, help="Training iterations for the smoke.")
    parser.add_argument("--top-n", type=int, default=1, help="Top-N segments to request during inference.")
    parser.add_argument(
        "--clear-selected-artifacts",
        action="store_true",
        help="Remove cached frame/feature artifacts for the selected videos before preprocessing.",
    )
    parser.add_argument(
        "--clear-visual-redis",
        action="store_true",
        help="Delete visual cache keys from Redis before preprocessing.",
    )
    parser.add_argument("--output", required=True, help="Path to write the JSON summary.")
    args = parser.parse_args()

    implementation_root = REPO_ROOT / "TGN" / "implementation"
    dataset_root = REPO_ROOT / "dataset"
    storage_root = implementation_root / "storage"

    if args.train_video_id and args.val_video_id and args.test_video_id:
        selected = (args.train_video_id, args.val_video_id, args.test_video_id)
    else:
        selected = choose_video_ids(dataset_root, seed=args.seed)
    train_video_id, val_video_id, test_video_id = selected

    wait_for_services(args.base_url, timeout_sec=args.wait_timeout)

    if args.clear_selected_artifacts:
        remove_paths(candidate_shared_paths(storage_root, [train_video_id, val_video_id, test_video_id]))
    if args.clear_visual_redis:
        clear_redis_patterns(["visual:preprocess:v1:*", "visual:features:v1:*"])

    summary: dict[str, Any] = {
        "tag": args.tag,
        "encoder": args.encoder,
        "selected_video_ids": {
            "train": train_video_id,
            "val": val_video_id,
            "test": test_video_id,
        },
        "phases": {},
    }

    rows: dict[str, dict[str, Any]] = {}

    def timed(name: str, fn):
        start = time.perf_counter()
        result = fn()
        summary["phases"][name] = {"duration_sec": round(time.perf_counter() - start, 3)}
        return result

    for split_name, video_id in [("train", train_video_id), ("val", val_video_id), ("test", test_video_id)]:
        preprocess_response = timed(
            f"preprocess_{split_name}",
            lambda video_id=video_id: json_request(
                "POST",
                f"{args.base_url}:{SERVICE_PORTS['visual-preprocessing-service']}/jobs/preprocess",
                payload={
                    "input_video_uri": f"file:///app/dataset/videos/{video_id}.avi",
                    "output_frame_size": args.output_frame_size,
                    "sample_every_sec": args.sample_every_sec,
                    "feature_service_url": args.feature_service_url,
                    "encoder": args.encoder,
                },
                timeout=1800,
            ),
        )
        summary.setdefault("preprocess", {})[split_name] = preprocess_response

        processed = timed(
            f"text_processing_{split_name}",
            lambda video_id=video_id: json_request(
                "POST",
                f"{args.base_url}:{SERVICE_PORTS['text-processing-service']}/jobs/process-aligned-text",
                payload={
                    "input_alignment_uri": f"file:///app/dataset/texts/{video_id}.aligned.tsv",
                    "artifact_uri": "shared://artifacts/text/v1",
                    "video_features_uri": f"shared://features/visual/{video_id}.vf.pt",
                    "base_name_prefix": f"{args.tag}-{video_id}",
                    "row_indices": [0],
                    "fps": args.fps,
                },
                timeout=600,
            ),
        )
        records = processed.get("records", [])
        if not records:
            raise RuntimeError(f"aligned-text processing produced no records for {video_id}")
        rows[split_name] = records[0]

    split_payloads = {
        "train": [
            {
                "base_name": rows["train"]["base_name"],
                "video_features_uri": rows["train"]["video_features_uri"],
                "text_processed_uri": rows["train"]["text_processed_uri"],
            }
        ],
        "val": [
            {
                "base_name": rows["val"]["base_name"],
                "video_features_uri": rows["val"]["video_features_uri"],
                "text_processed_uri": rows["val"]["text_processed_uri"],
            }
        ],
        "test": [dict(rows["test"], video_id=test_video_id)],
    }

    split_paths = {
        split_name: storage_root / "splits" / split_name / f"{args.tag}.json"
        for split_name in ["train", "val", "test"]
    }
    for split_name, payload in split_payloads.items():
        split_paths[split_name].parent.mkdir(parents=True, exist_ok=True)
        split_paths[split_name].write_text(json.dumps(payload, indent=2) + "\n")

    training_response = timed(
        "training",
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
                    batch_size=1,
                    log_every=1,
                    valid_niter=1,
                ),
            },
            timeout=3600,
        ),
    )
    inference_response = timed(
        "inference",
        lambda: json_request(
            "POST",
            f"{args.base_url}:{SERVICE_PORTS['inference-service']}/infer/ground",
            payload={
                "model_uri": shared_uri("models", f"{args.tag}.bin"),
                "video_features_uri": rows["test"]["video_features_uri"],
                "text_processed_uri": rows["test"]["text_processed_uri"],
                "top_n": args.top_n,
            },
            timeout=1200,
        ),
    )
    evaluation_response = timed(
        "evaluation",
        lambda: json_request(
            "POST",
            f"{args.base_url}:{SERVICE_PORTS['evaluation-service']}/jobs/evaluate",
            payload={
                "job": {
                    "job_id": args.tag,
                    "trace_id": f"{args.tag}-trace",
                    "dataset": "smoke",
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "version": "v1",
                },
                "model_uri": shared_uri("models", f"{args.tag}.bin"),
                "test_split_uri": shared_uri("splits", "test", f"{args.tag}.json"),
                "features_root_uri": "shared://features/visual",
                "metrics": ["R@1_IOU0.5"],
            },
            timeout=3600,
        ),
    )

    summary["records"] = rows
    summary["training_response"] = training_response
    summary["inference_response"] = inference_response
    summary["evaluation_response"] = evaluation_response
    summary["split_paths"] = {name: str(path) for name, path in split_paths.items()}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
