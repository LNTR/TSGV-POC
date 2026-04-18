#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "TGN-smoke"))

from run_compose_dataset_workflow import (  # noqa: E402
    SERVICE_PORTS,
    candidate_shared_paths,
    clear_redis_patterns,
    json_request,
    remove_paths,
    wait_for_services,
)


def timed_request(method: str, url: str, payload: dict[str, Any], timeout: int) -> tuple[float, dict[str, Any]]:
    start = time.perf_counter()
    response = json_request(method, url, payload=payload, timeout=timeout)
    return round(time.perf_counter() - start, 3), response


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure repeated TGN preprocess and text-processing reuse behavior.")
    parser.add_argument("--base-url", default="http://127.0.0.1", help="Base host for published service ports.")
    parser.add_argument(
        "--feature-service-url",
        default="http://visual-feature-extraction-service:8002/jobs/features",
        help="Internal URL used by visual-preprocessing-service to reach feature extraction.",
    )
    parser.add_argument("--video-id", default="s27-d34", help="Video id used for preprocess reuse checks.")
    parser.add_argument(
        "--artifact-uri",
        default="shared://artifacts/text/v1",
        help="Text artifact directory used for text-processing reuse checks.",
    )
    parser.add_argument("--output", required=True, help="Path to write the reuse JSON summary.")
    args = parser.parse_args()

    storage_root = REPO_ROOT / "TGN" / "implementation" / "storage"
    text_input_path = storage_root / "text" / "raw" / "test_inference" / "reuse-check.txt"
    text_input_path.parent.mkdir(parents=True, exist_ok=True)
    text_input_path.write_text("take out ginger from counter cupboard using hand\n")

    wait_for_services(args.base_url, timeout_sec=120)

    remove_paths(candidate_shared_paths(storage_root, [args.video_id]))
    clear_redis_patterns(["visual:preprocess:v1:*", "visual:features:v1:*", "pred:text:v1:*", "pred:text:v1:index:*"])

    preprocess_payload = {
        "input_video_uri": f"file:///app/dataset/videos/{args.video_id}.avi",
        "output_frame_size": 256,
        "sample_every_sec": 3.0,
        "feature_service_url": args.feature_service_url,
    }
    preprocess_url = f"{args.base_url}:{SERVICE_PORTS['visual-preprocessing-service']}/jobs/preprocess"
    first_preprocess_sec, first_preprocess = timed_request("POST", preprocess_url, preprocess_payload, 1800)
    second_preprocess_sec, second_preprocess = timed_request("POST", preprocess_url, preprocess_payload, 1800)

    process_text_payload = {
        "input_text_uri": "shared://text/raw/test_inference/reuse-check.txt",
        "artifact_uri": args.artifact_uri,
        "base_name": "reuse-check",
        "video_features_uri": f"shared://features/visual/{args.video_id}.vf.pt",
    }
    process_text_url = f"{args.base_url}:{SERVICE_PORTS['text-processing-service']}/jobs/process-text"
    first_text_sec, first_text = timed_request("POST", process_text_url, process_text_payload, 600)
    second_text_sec, second_text = timed_request("POST", process_text_url, process_text_payload, 600)

    summary = {
        "status": "complete",
        "preprocess": {
            "video_id": args.video_id,
            "first_duration_sec": first_preprocess_sec,
            "second_duration_sec": second_preprocess_sec,
            "first_cache_hit": bool(first_preprocess.get("metadata", {}).get("cache_hit")),
            "second_cache_hit": bool(second_preprocess.get("metadata", {}).get("cache_hit")),
            "first_forwarded_cache_hit": bool(first_preprocess.get("forwarded", {}).get("metadata", {}).get("cache_hit")),
            "second_forwarded_cache_hit": bool(second_preprocess.get("forwarded", {}).get("metadata", {}).get("cache_hit")),
            "first_response": first_preprocess,
            "second_response": second_preprocess,
            "second_faster_bool": second_preprocess_sec < first_preprocess_sec,
        },
        "text_processing": {
            "input_text_uri": process_text_payload["input_text_uri"],
            "first_duration_sec": first_text_sec,
            "second_duration_sec": second_text_sec,
            "first_cache_hit": bool(first_text.get("metadata", {}).get("cache_hit")),
            "second_cache_hit": bool(second_text.get("metadata", {}).get("cache_hit")),
            "first_response": first_text,
            "second_response": second_text,
            "second_faster_bool": second_text_sec < first_text_sec,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
