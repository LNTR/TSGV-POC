from __future__ import annotations

import sys
from pathlib import Path

import torch

IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[2]
if str(IMPLEMENTATION_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPLEMENTATION_ROOT))

from tall_model import CTRLTemporalLocalizer
from tall_runtime import (
    build_candidate_windows,
    decode_predictions,
    load_text_processed,
    load_visual_features,
    resolve_uri_to_path,
    sentence_embedding_from_text_processed,
)


def infer_segments(
    model_uri: str,
    video_features_uri: str,
    text_processed_uri: str,
    top_n: int,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CTRLTemporalLocalizer.load(resolve_uri_to_path(model_uri), map_location=device)
    model = model.to(device)
    model.eval()

    visual_features = load_visual_features(video_features_uri)
    text_processed = load_text_processed(text_processed_uri)
    sentence_embedding = sentence_embedding_from_text_processed(text_processed)

    candidate_features, anchor_times = build_candidate_windows(
        visual_features,
        max_window_scales=model.config.max_window_scales,
        context_size=model.config.context_size,
        sample_every_sec=model.config.sample_every_sec,
    )
    if candidate_features.shape[0] == 0:
        return []
    if int(candidate_features.shape[1]) != int(model.config.visual_feature_size):
        raise ValueError(
            "candidate feature dimension mismatch: "
            f"artifact={candidate_features.shape[1]} model={model.config.visual_feature_size}"
        )

    with torch.no_grad():
        logits, start_offsets, end_offsets = model(
            candidate_features.to(device),
            sentence_embedding.to(device),
        )

    return decode_predictions(
        logits=logits,
        start_offsets=start_offsets,
        end_offsets=end_offsets,
        anchor_times=anchor_times,
        duration_sec=float(visual_features.shape[0] * model.config.sample_every_sec),
        top_n=top_n,
        nms_threshold=model.config.nms_threshold,
    )
