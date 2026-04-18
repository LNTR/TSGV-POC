#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from run_compose_dataset_workflow import (
    COMPOSE_SERVICES,
    PHASE_SERVICES,
    SERVICE_PORTS,
    build_hyperparams,
    candidate_shared_paths,
    cleanup_shared_paths,
    cleanup_text_processed_uris,
    clear_redis_patterns,
    copy_service_file_if_exists,
    discover_video_ids,
    ensure_feature_extraction_used_gpu,
    finalize_phase_stats,
    json_request,
    remove_paths,
    run_compose,
    run_timed_phase,
    shared_uri,
    wait_for_services,
    write_timeline_csv,
)


def choose_smoke_video_ids(dataset_root: Path, seed: int) -> tuple[str, str, str]:
    video_ids = discover_video_ids(dataset_root)
    shuffled = list(video_ids)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 3:
        raise RuntimeError("need at least 3 matching dataset items for the smoke test")
    return shuffled[0], shuffled[1], shuffled[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a small 3-video TGN Docker Compose smoke test for preprocess, text processing, training, inference, and evaluation."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1", help="Base host for published Compose ports.")
    parser.add_argument(
        "--feature-service-url",
        default="http://visual-feature-extraction-service:8002",
        help="Internal URL that the preprocessing container should use to reach the feature service.",
    )
    parser.add_argument("--tag", default="compose_smoke", help="Run folder name under TGN-smoke/.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed used to choose the 3 smoke videos.")
    parser.add_argument("--output-frame-size", type=int, default=224, help="Frame size sent to preprocessing.")
    parser.add_argument("--sample-every-sec", type=float, default=5.0, help="Temporal sampling rate for preprocessing.")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS used when converting aligned rows to seconds.")
    parser.add_argument("--max-iter", type=int, default=4, help="Training iterations for the smoke run.")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--log-every", type=int, default=1, help="Training logging interval.")
    parser.add_argument("--valid-niter", type=int, default=1, help="Validation interval in training iterations.")
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
        help="Require visual feature extraction to report a CUDA device; fail the smoke run otherwise.",
    )
    parser.add_argument(
        "--force-feature-recompute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete cached frame/feature artifacts and clear Redis visual cache keys before preprocessing so extraction runs fresh.",
    )
    parser.add_argument("--cleanup", action="store_true", help="Delete generated smoke artifacts before exiting.")
    parser.add_argument(
        "--compose-managed",
        action="store_true",
        help="Start the TGN GPU Docker Compose services before the smoke run and stop them afterward.",
    )
    parser.add_argument(
        "--compose-no-build",
        action="store_true",
        help="When used with --compose-managed, skip rebuilding and start from whatever images already exist locally.",
    )
    parser.add_argument(
        "--keep-compose-up",
        action="store_true",
        help="When used with --compose-managed, leave the Compose services running after the smoke run.",
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

    train_video_id, val_video_id, test_video_id = choose_smoke_video_ids(dataset_root, seed=args.seed)
    selected_video_ids = [train_video_id, val_video_id, test_video_id]
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

        for split_name, video_id in [("train", train_video_id), ("val", val_video_id), ("test", test_video_id)]:
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
                        "row_indices": [0],
                        "fps": args.fps,
                    },
                    timeout=600,
                ),
            )

            records = processed.get("records", [])
            if not records:
                raise RuntimeError(f"aligned-text processing produced no records for {video_id}")

            record = records[0]
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
                sample_inference_record = evaluation_record

            metadata_path = storage_root / "features" / "metadata" / f"{video_id}.vf.meta.json"
            if video_id not in feature_devices and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                feature_devices[video_id] = metadata.get("device_used", "unknown")

        train_split_text = json.dumps(split_records["train"], indent=2) + "\n"
        val_split_text = json.dumps(split_records["val"], indent=2) + "\n"
        test_split_text = json.dumps(evaluation_records, indent=2) + "\n"
        train_split_path.parent.mkdir(parents=True, exist_ok=True)
        val_split_path.parent.mkdir(parents=True, exist_ok=True)
        test_split_path.parent.mkdir(parents=True, exist_ok=True)
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

        if sample_inference_record is None:
            raise RuntimeError("smoke test did not produce a sample inference record")

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
                        "dataset": "smoke",
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
            "selected_video_ids": {
                "train": train_video_id,
                "val": val_video_id,
                "test": test_video_id,
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
                "base_name": sample_inference_record["base_name"],
                "query_text": sample_inference_record.get("query_text"),
                "segments": sample_inference_response.get("segments", []),
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
