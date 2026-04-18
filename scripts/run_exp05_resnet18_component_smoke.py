#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = ROOT / "TGN" / "implementation" / "storage"
FRAME_DIR = STORAGE_ROOT / "frames" / "processed"
FRAME_META_DIR = STORAGE_ROOT / "frames" / "metadata"
ALIGN_ROOT = STORAGE_ROOT / "smoke" / "generated_alignments"
TEXT_SPLIT_ROOT = STORAGE_ROOT / "splits"

SERVICES = {
    "feature": "http://127.0.0.1:8002/jobs/features",
    "text": "http://127.0.0.1:8003/jobs/process-aligned-text",
    "train": "http://127.0.0.1:8004/jobs/train",
    "infer": "http://127.0.0.1:8005/infer/ground",
    "evaluate": "http://127.0.0.1:8006/jobs/evaluate",
}

VIDEOS = {
    "train": {
        "video_id": "s13-d21",
        "start_frame": 252,
        "end_frame": 452,
        "action": "take out",
        "agent": "hand",
        "object": "cutting board",
        "location": "counter,drawer",
    },
    "val": {
        "video_id": "s22-d25",
        "start_frame": 290,
        "end_frame": 455,
        "action": "take out",
        "agent": "hand",
        "object": "plastic paper bag",
        "location": "counter,cupboard",
    },
    "test": {
        "video_id": "s27-d34",
        "start_frame": 233,
        "end_frame": 435,
        "action": "take out",
        "agent": "hand",
        "object": "ginger",
        "location": "counter,cupboard",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the exp05 resnet18 component smoke.")
    parser.add_argument(
        "--output",
        default="TGN-smoke/exp05_resnet18_component_smoke/summary.json",
        help="Summary JSON output path.",
    )
    parser.add_argument(
        "--tag",
        default="exp05-resnet18-component",
        help="Artifact prefix used for this smoke run.",
    )
    return parser.parse_args()


def timed_post(url: str, payload: dict[str, Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    response = requests.post(url, json=payload, timeout=300)
    duration = round(time.perf_counter() - start, 3)
    response.raise_for_status()
    return duration, response.json()


def copy_frame_artifacts(source_video: str, target_base: str) -> tuple[str, str]:
    source_frames = FRAME_DIR / f"{source_video}.vp.npy"
    source_meta = FRAME_META_DIR / f"{source_video}.vp.meta.json"
    target_frames = FRAME_DIR / f"{target_base}.vp.npy"
    target_meta = FRAME_META_DIR / f"{target_base}.vp.meta.json"
    if not source_frames.exists() or not source_meta.exists():
        raise FileNotFoundError(f"missing cached frame artifacts for {source_video}")
    shutil.copyfile(source_frames, target_frames)
    metadata = json.loads(source_meta.read_text())
    metadata["base_name"] = target_base
    metadata["frames_uri"] = f"shared://frames/processed/{target_base}.vp.npy"
    metadata["metadata_uri"] = f"shared://frames/metadata/{target_base}.vp.meta.json"
    target_meta.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata["frames_uri"], metadata["metadata_uri"]


def write_alignment_file(split_name: str, spec: dict[str, Any]) -> dict[str, Any]:
    ALIGN_ROOT.mkdir(parents=True, exist_ok=True)
    path = ALIGN_ROOT / f"{spec['video_id']}-{split_name}.aligned.tsv"
    line = "\t".join(
        [
            str(spec["start_frame"]),
            str(spec["end_frame"]),
            spec["action"],
            spec["agent"],
            spec["object"],
            spec["location"],
        ]
    )
    path.write_text(line + "\n", encoding="utf-8")
    query_text = f"{spec['action']} {spec['object']} from {spec['location']} using {spec['agent']}"
    return {
        "path": path,
        "uri": f"shared://smoke/generated_alignments/{path.name}",
        "row_index": 0,
        "start_frame": spec["start_frame"],
        "end_frame": spec["end_frame"],
        "query_text": query_text,
        "start_time": spec["start_frame"] / 30.0,
        "end_time": spec["end_frame"] / 30.0,
    }


def write_split(tag: str, split_name: str, record: dict[str, Any]) -> str:
    path = TEXT_SPLIT_ROOT / split_name / f"{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([record], indent=2) + "\n")
    return f"shared://splits/{split_name}/{tag}.json"


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    phase_timings: dict[str, dict[str, float]] = {}
    feature_responses: dict[str, Any] = {}
    text_records: dict[str, Any] = {}
    split_uris: dict[str, str] = {}

    for split_name, spec in VIDEOS.items():
        video_id = spec["video_id"]
        target_base = f"{args.tag}-{video_id}"
        frames_uri, _ = copy_frame_artifacts(video_id, target_base)
        duration, feature_response = timed_post(
            SERVICES["feature"],
            {
                "frames_uri": frames_uri,
                "encoder": "resnet18",
            },
        )
        phase_timings[f"feature_extraction_{split_name}"] = {"duration_sec": duration}
        feature_responses[split_name] = feature_response

        row = write_alignment_file(split_name, spec)
        duration, text_response = timed_post(
            SERVICES["text"],
            {
                "input_alignment_uri": row["uri"],
                "artifact_uri": "shared://artifacts/text/v1",
                "video_features_uri": feature_response["metadata"]["features_uri"],
                "row_indices": [0],
                "base_name_prefix": target_base,
            },
        )
        phase_timings[f"text_processing_{split_name}"] = {"duration_sec": duration}
        item = text_response["records"][0]
        text_records[split_name] = {
            "base_name": item["base_name"],
            "text_processed_uri": item["text_processed_uri"],
            "video_features_uri": item["video_features_uri"],
            "row_index": item["row_index"],
            "query_text": item["query_text"],
            "start_frame": item["start_frame"],
            "end_frame": item["end_frame"],
            "start_time": item["start_time"],
            "end_time": item["end_time"],
            "num_token_ids": item["num_token_ids"],
        }
        split_uris[split_name] = write_split(args.tag, split_name, item)

    duration, training_response = timed_post(
        SERVICES["train"],
        {
            "train_split_uri": split_uris["train"],
            "val_split_uri": split_uris["val"],
            "features_root_uri": "shared://features/visual",
            "output_model_uri": f"shared://models/{args.tag}.bin",
            "hyperparams": {
                "K": 4,
                "delta": 2,
                "threshold": 0.5,
                "batch_size": 1,
                "lr": 0.001,
                "hidden_size_textual_lstm": 64,
                "hidden_size_visual_lstm": 64,
                "hidden_size_ilstm": 64,
                "word_embed_size": 50,
                "visual_feature_size": 4096,
                "log_every": 1,
                "max_iter": 1,
                "valid_niter": 1,
                "top_n_eval": 1,
                "patience": 1,
                "max_num_trial": 1,
                "lr_decay": 0.5,
                "fps": 30,
                "sample_rate": 150,
            },
        },
    )
    phase_timings["training"] = {"duration_sec": duration}

    duration, inference_response = timed_post(
        SERVICES["infer"],
        {
            "model_uri": f"shared://models/{args.tag}.bin",
            "video_features_uri": text_records["test"]["video_features_uri"],
            "text_processed_uri": text_records["test"]["text_processed_uri"],
            "top_n": 1,
        },
    )
    phase_timings["inference"] = {"duration_sec": duration}

    duration, evaluation_response = timed_post(
        SERVICES["evaluate"],
        {
            "job": {
                "job_id": args.tag,
                "trace_id": f"{args.tag}-trace",
                "dataset": "smoke",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "version": "v1",
            },
            "model_uri": f"shared://models/{args.tag}.bin",
            "test_split_uri": split_uris["test"],
            "features_root_uri": "shared://features/visual",
            "metrics": ["R@1_IOU0.5"],
        },
    )
    phase_timings["evaluation"] = {"duration_sec": duration}

    summary = {
        "tag": args.tag,
        "encoder": "resnet18",
        "selected_video_ids": {key: value["video_id"] for key, value in VIDEOS.items()},
        "phases": phase_timings,
        "feature_responses": feature_responses,
        "records": text_records,
        "training_response": training_response,
        "inference_response": inference_response,
        "evaluation_response": evaluation_response,
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
