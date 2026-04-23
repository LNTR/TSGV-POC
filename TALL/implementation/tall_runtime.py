from __future__ import annotations

import json
import os
from functools import lru_cache

import torch

from tall_text import (
    TextArtifact,
    load_text_artifact,
    resolve_uri_to_path,
    sentence_embedding_from_text_processed,
)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def load_text_processed(text_processed_uri: str) -> dict:
    text_processed_path = resolve_uri_to_path(text_processed_uri)
    if not os.path.exists(text_processed_path):
        raise FileNotFoundError(f"text processed file not found: {text_processed_path}")

    data = load_json(text_processed_path)
    if not isinstance(data, dict):
        raise ValueError(f"text processed file must contain a JSON object: {text_processed_path}")
    if "artifact_uri" not in data:
        raise ValueError(f"text processed file must contain 'artifact_uri': {text_processed_path}")
    if "sentence_embedding" not in data:
        raise ValueError(f"text processed file must contain 'sentence_embedding': {text_processed_path}")
    return data


@lru_cache(maxsize=512)
def load_visual_features(video_features_uri: str) -> torch.Tensor:
    video_features_path = resolve_uri_to_path(video_features_uri)
    if not os.path.exists(video_features_path):
        raise FileNotFoundError(f"video features not found: {video_features_path}")

    tensor = torch.load(video_features_path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"video features must be a torch.Tensor: {video_features_path}")
    if tensor.ndim != 2:
        raise ValueError(f"video features must have shape [T, feature_dim]: {video_features_path}")
    return tensor.to(torch.float32)


def build_candidate_windows(
    frame_features: torch.Tensor,
    max_window_scales: int,
    context_size: int,
    sample_every_sec: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if frame_features.ndim != 2:
        raise ValueError("frame_features must have shape [T, feature_dim]")

    timesteps, feature_dim = frame_features.shape
    if timesteps == 0:
        return (
            torch.zeros((0, feature_dim * 3), dtype=frame_features.dtype),
            torch.zeros((0, 2), dtype=frame_features.dtype),
        )

    features = []
    anchors = []
    max_scales = max(1, min(int(max_window_scales), timesteps))
    context_width = max(1, int(context_size))

    for scale in range(1, max_scales + 1):
        for start_idx in range(0, timesteps - scale + 1):
            end_idx = start_idx + scale - 1
            center = frame_features[start_idx:end_idx + 1].mean(dim=0)

            left_slice = frame_features[max(0, start_idx - context_width):start_idx]
            right_slice = frame_features[end_idx + 1:min(timesteps, end_idx + 1 + context_width)]

            left = left_slice.mean(dim=0) if left_slice.numel() else center
            right = right_slice.mean(dim=0) if right_slice.numel() else center

            features.append(torch.cat([left, center, right], dim=0))
            anchors.append(
                [
                    float(start_idx * sample_every_sec),
                    float((end_idx + 1) * sample_every_sec),
                ]
            )

    return torch.stack(features), torch.tensor(anchors, dtype=frame_features.dtype)


def compute_iou(interval_a: tuple[float, float], interval_b: tuple[float, float]) -> float:
    start = max(float(interval_a[0]), float(interval_b[0]))
    end = min(float(interval_a[1]), float(interval_b[1]))
    intersection = max(0.0, end - start)

    union_start = min(float(interval_a[0]), float(interval_b[0]))
    union_end = max(float(interval_a[1]), float(interval_b[1]))
    union = max(0.0, union_end - union_start)
    if union <= 0.0:
        return 0.0
    return intersection / union


def temporal_nms(segments: list[dict], threshold: float) -> list[dict]:
    selected: list[dict] = []
    for segment in sorted(segments, key=lambda item: float(item["score"]), reverse=True):
        overlaps = [
            compute_iou(
                (segment["start_sec"], segment["end_sec"]),
                (chosen["start_sec"], chosen["end_sec"]),
            )
            for chosen in selected
        ]
        if all(overlap <= threshold for overlap in overlaps):
            selected.append(segment)
    return selected


def decode_predictions(
    logits: torch.Tensor,
    start_offsets: torch.Tensor,
    end_offsets: torch.Tensor,
    anchor_times: torch.Tensor,
    duration_sec: float,
    top_n: int,
    nms_threshold: float,
) -> list[dict]:
    if top_n <= 0 or logits.numel() == 0:
        return []

    scores = torch.sigmoid(logits).detach().cpu()
    start_offsets = start_offsets.detach().cpu()
    end_offsets = end_offsets.detach().cpu()
    anchor_times = anchor_times.detach().cpu()

    segments = []
    for idx in range(scores.shape[0]):
        anchor_start = float(anchor_times[idx, 0].item())
        anchor_end = float(anchor_times[idx, 1].item())
        start_sec = min(max(anchor_start + float(start_offsets[idx].item()), 0.0), duration_sec)
        end_sec = min(max(anchor_end + float(end_offsets[idx].item()), start_sec), duration_sec)
        segments.append(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "score": float(scores[idx].item()),
            }
        )

    ranked = sorted(segments, key=lambda item: item["score"], reverse=True)
    if nms_threshold > 0.0:
        ranked = temporal_nms(ranked, nms_threshold)
    return ranked[:top_n]
