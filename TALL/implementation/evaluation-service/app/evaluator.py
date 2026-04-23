import importlib
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import requests
import torch


def resolve_uri_to_path(uri: str) -> str:
    storage_root = Path(
        os.environ.get(
            "IMPLEMENTATION_STORAGE_ROOT",
            Path(__file__).resolve().parents[2] / "storage",
        )
    ).resolve()

    if uri.startswith("file://"):
        return urlparse(uri).path
    if uri.startswith("shared://"):
        return str((storage_root / uri[len("shared://"):].lstrip("/")).resolve())

    path = Path(uri)
    if not path.is_absolute():
        return str((storage_root / path).resolve())
    return str(path)


def load_json(uri: str) -> list | dict:
    with open(resolve_uri_to_path(uri)) as f:
        return json.load(f)


def compute_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    if end_a < start_b or end_b < start_a:
        return 0.0

    overlap_start = max(start_a, start_b)
    overlap_end = min(end_a, end_b)
    return max(0.0, overlap_end - overlap_start)


def parse_metric(metric: str) -> tuple[int, float]:
    match = re.fullmatch(r"R@(\d+)_IOU([0-9]*\.?[0-9]+)", metric)
    if match is None:
        raise ValueError(f"unsupported metric format: {metric}")
    return int(match.group(1)), float(match.group(2))


def score_prediction_records(records: list[dict], metrics: list[str]) -> dict[str, float]:
    if not records:
        return {metric: 0.0 for metric in metrics}

    scores: dict[str, float] = {}
    for metric in metrics:
        try:
            n_top, threshold = parse_metric(metric)
        except ValueError:
            scores[metric] = 0.0
            continue

        hits = 0
        for item in records:
            gold_start = float(item["gold_start"])
            gold_end = float(item["gold_end"])
            preds = item.get("pred_segments", [])
            preds = sorted(preds, key=lambda p: float(p.get("score", 0.0)), reverse=True)[:n_top]

            max_overlap = 0.0
            for pred in preds:
                overlap = compute_overlap(
                    float(pred.get("start", 0.0)),
                    float(pred.get("end", 0.0)),
                    gold_start,
                    gold_end,
                )
                max_overlap = max(max_overlap, overlap)

            hits += int(max_overlap > threshold)

        scores[metric] = float(hits / len(records))

    return scores


def derive_features_uri(base_name: str, features_root_uri: str) -> str:
    base = features_root_uri.rstrip("/")
    if base.endswith(".vf.pt"):
        return base
    return f"{base}/{base_name}.vf.pt"


def resolve_text_processed_uri(record: dict) -> str:
    candidates = [
        record.get("text_processed_uri"),
        record.get("inference_text_processed_uri"),
        record.get("train_text_processed_uri"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    raise ValueError(f"record is missing a text processed URI: {record}")


def resolve_video_features_uri(record: dict, features_root_uri: str) -> str:
    value = record.get("video_features_uri")
    if isinstance(value, str) and value:
        return value

    base_name = record.get("base_name") or record.get("video_id")
    if not isinstance(base_name, str) or not base_name:
        raise ValueError(f"record is missing both video_features_uri and base_name/video_id: {record}")

    return derive_features_uri(base_name=base_name, features_root_uri=features_root_uri)


def resolve_gold_times(record: dict, text_processed_uri: str | None = None) -> tuple[float, float]:
    start = record.get("gold_start", record.get("start_time"))
    end = record.get("gold_end", record.get("end_time"))
    if start is not None and end is not None:
        return float(start), float(end)

    if text_processed_uri is None:
        raise ValueError(f"record is missing gold times: {record}")

    payload = load_json(text_processed_uri)
    if not isinstance(payload, dict):
        raise ValueError(f"text processed payload must be a JSON object: {text_processed_uri}")
    start = payload.get("start_time")
    end = payload.get("end_time")
    if start is None or end is None:
        raise ValueError(f"text processed payload is missing start/end_time: {text_processed_uri}")
    return float(start), float(end)


@contextmanager
def inference_import_path():
    inference_root = Path(__file__).resolve().parents[2] / "inference-service"
    inserted = False
    if inference_root.exists():
        sys.path.insert(0, str(inference_root))
        inserted = True
    try:
        yield
    finally:
        if inserted and sys.path and sys.path[0] == str(inference_root):
            sys.path.pop(0)


def infer_segments_locally(
    model_uri: str,
    video_features_uri: str,
    text_processed_uri: str,
    top_n: int,
) -> list[dict]:
    with inference_import_path():
        pipeline = importlib.import_module("app.pipeline")
        segments = pipeline.infer_segments(
            model_uri=model_uri,
            video_features_uri=video_features_uri,
            text_processed_uri=text_processed_uri,
            top_n=top_n,
        )
    return [
        {
            "start": float(segment["start_sec"]),
            "end": float(segment["end_sec"]),
            "score": float(segment["score"]),
        }
        for segment in segments
    ]


def infer_segments_via_service(
    model_uri: str,
    video_features_uri: str,
    text_processed_uri: str,
    top_n: int,
) -> list[dict]:
    inference_service_url = os.environ.get("INFERENCE_SERVICE_URL", "").strip()
    if not inference_service_url:
        raise RuntimeError("INFERENCE_SERVICE_URL is not configured for evaluation-time inference.")

    response = requests.post(
        f"{inference_service_url.rstrip('/')}/infer/ground",
        json={
            "model_uri": model_uri,
            "video_features_uri": video_features_uri,
            "text_processed_uri": text_processed_uri,
            "top_n": top_n,
        },
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()
    segments = body.get("segments", [])
    return [
        {
            "start": float(segment["start_sec"]),
            "end": float(segment["end_sec"]),
            "score": float(segment["score"]),
        }
        for segment in segments
    ]


def build_prediction_records(
    model_uri: str,
    test_split_uri: str,
    features_root_uri: str,
    max_top_n: int,
) -> list[dict]:
    raw_records = load_json(test_split_uri)
    if not isinstance(raw_records, list):
        raise ValueError("evaluation input must be a JSON list.")
    if not raw_records:
        return []

    first = raw_records[0]
    if not isinstance(first, dict):
        raise ValueError("evaluation records must be JSON objects.")

    if "pred_segments" in first and "gold_start" in first and "gold_end" in first:
        return raw_records

    prediction_records = []
    for record in raw_records:
        if not isinstance(record, dict):
            raise ValueError("evaluation records must be JSON objects.")

        text_processed_uri = resolve_text_processed_uri(record)
        video_features_uri = resolve_video_features_uri(record, features_root_uri=features_root_uri)
        gold_start, gold_end = resolve_gold_times(record, text_processed_uri=text_processed_uri)

        try:
            pred_segments = infer_segments_locally(
                model_uri=model_uri,
                video_features_uri=video_features_uri,
                text_processed_uri=text_processed_uri,
                top_n=max_top_n,
            )
        except ModuleNotFoundError:
            pred_segments = infer_segments_via_service(
                model_uri=model_uri,
                video_features_uri=video_features_uri,
                text_processed_uri=text_processed_uri,
                top_n=max_top_n,
            )

        prediction_records.append(
            {
                "base_name": record.get("base_name"),
                "video_id": record.get("video_id"),
                "query_text": record.get("query_text"),
                "gold_start": gold_start,
                "gold_end": gold_end,
                "pred_segments": pred_segments,
            }
        )

    return prediction_records


def run_evaluation_job(
    model_uri: str,
    test_split_uri: str,
    metrics: list[str],
    features_root_uri: str,
) -> dict[str, float]:
    model_path = resolve_uri_to_path(model_uri)
    test_split_path = resolve_uri_to_path(test_split_uri)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")
    if not os.path.exists(test_split_path):
        raise FileNotFoundError(f"test split not found: {test_split_path}")

    _ = torch.load(model_path, map_location="cpu")

    parsed_metrics = []
    for metric in metrics:
        try:
            parsed_metrics.append((metric, *parse_metric(metric)))
        except ValueError:
            parsed_metrics.append((metric, 0, 0.0))

    max_top_n = max((n_top for _, n_top, _ in parsed_metrics), default=0)
    prediction_records = build_prediction_records(
        model_uri=model_uri,
        test_split_uri=test_split_uri,
        features_root_uri=features_root_uri,
        max_top_n=max_top_n,
    )
    return score_prediction_records(prediction_records, metrics)
