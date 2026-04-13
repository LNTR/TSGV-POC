import json
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch

from .model.tgn import TGN


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


@lru_cache(maxsize=16)
def _load_text_artifact_cached(artifact_path: str):
    root = Path(artifact_path)
    vocab_path = root / "vocab.json"
    embeddings_path = root / "embeddings.npy"

    if not vocab_path.exists():
        raise FileNotFoundError(f"missing vocab file: {vocab_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"missing embeddings file: {embeddings_path}")

    raw_vocab = json.loads(vocab_path.read_text())
    if isinstance(raw_vocab, dict):
        word2id = {str(k): int(v) for k, v in raw_vocab.items()}
    elif isinstance(raw_vocab, list):
        word2id = {str(word): idx for idx, word in enumerate(raw_vocab)}
    else:
        raise ValueError("vocab.json must be a dict[word->id] or list[word].")

    embedding_matrix = np.load(embeddings_path)
    if embedding_matrix.ndim != 2:
        raise ValueError("embeddings.npy must be rank-2 [vocab_size, embed_dim].")

    return {
        "word2id": word2id,
        "embedding_matrix": embedding_matrix,
    }


def _embed_from_token_ids(
    token_ids: list[int],
    embedding_matrix: np.ndarray,
    unk_id: int,
) -> torch.Tensor:
    if not token_ids:
        token_ids = [unk_id]

    safe_ids = []
    max_id = embedding_matrix.shape[0] - 1
    for tid in token_ids:
        if isinstance(tid, bool):
            tid = int(tid)
        if not isinstance(tid, int):
            tid = unk_id
        if tid < 0 or tid > max_id:
            tid = unk_id
        safe_ids.append(tid)

    arr = embedding_matrix[np.array(safe_ids, dtype=np.int64)]
    return torch.from_numpy(arr).to(torch.float32).unsqueeze(0)


def _load_text_processed(text_processed_uri: str) -> dict:
    text_processed_path = resolve_uri_to_path(text_processed_uri)
    with open(text_processed_path) as f:
        data = json.load(f)

    if "artifact_uri" not in data:
        raise ValueError("text processed file must contain 'artifact_uri'.")
    if "token_ids" not in data:
        raise ValueError("text processed file must contain 'token_ids'.")

    return data


def infer_segments(
    model_uri: str,
    video_features_uri: str,
    text_processed_uri: str,
    top_n: int,
    fps: int = 30,
    sample_rate: int = 150,
    delta: int = 2,
):
    model_path = resolve_uri_to_path(model_uri)
    features_path = resolve_uri_to_path(video_features_uri)
    text_processed = _load_text_processed(text_processed_uri)
    artifact_path = resolve_uri_to_path(text_processed["artifact_uri"])

    artifact = _load_text_artifact_cached(artifact_path)

    model = TGN.load(model_path)
    model.eval()

    embedding_matrix = artifact["embedding_matrix"]
    if embedding_matrix.shape[1] != model.word_embed_size:
        raise ValueError(
            "embedding dimension mismatch: "
            f"artifact={embedding_matrix.shape[1]} model={model.word_embed_size}"
        )

    word2id = artifact["word2id"]
    unk_id = int(word2id.get("<unk>", 1))

    visual_feature = torch.load(features_path, map_location="cpu")
    text_embed = _embed_from_token_ids(text_processed["token_ids"], embedding_matrix, unk_id=unk_id)

    length_t = int(text_processed.get("length_t", 0))
    if length_t <= 0:
        length_t = text_embed.shape[1]
    length_t = min(length_t, text_embed.shape[1])

    with torch.no_grad():
        probs, mask = model(features_v=[visual_feature], textual_input=text_embed, lengths_t=[length_t])

    scores = (probs * mask).view(-1)
    if scores.numel() == 0:
        return []

    k_scales = probs.shape[2]
    k = min(top_n, scores.numel())
    values, indices = torch.topk(scores, k=k)
    video_duration_sec = float(visual_feature.shape[0] * sample_rate / fps)

    segments = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        end_time = (idx // k_scales) * sample_rate / fps
        scale_num = (idx % k_scales) + 1
        start_time = end_time - (scale_num * delta * sample_rate / fps)
        end_time = min(max(float(end_time), 0.0), video_duration_sec)
        start_time = min(max(float(start_time), 0.0), end_time)
        segments.append({"start_sec": start_time, "end_sec": end_time, "score": float(score)})

    return segments
