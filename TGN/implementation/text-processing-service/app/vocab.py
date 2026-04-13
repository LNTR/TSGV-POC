import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import numpy as np
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


def pad_textual_data(sents: List[List[int]], pad_token: int) -> List[List[int]]:
    if not sents:
        return []
    longest = max(len(sent) for sent in sents)
    return [sent + [pad_token] * (longest - len(sent)) for sent in sents]


@dataclass
class TextArtifact:
    word2id: Dict[str, int]
    embedding_matrix: np.ndarray


class Vocab:
    def __init__(self, word2id: Dict[str, int]):
        self.word2id = dict(word2id)
        if "<pad>" not in self.word2id or "<unk>" not in self.word2id:
            raise ValueError("vocab must include '<pad>' and '<unk>' tokens.")

    def __getitem__(self, word: str) -> int:
        return self.word2id.get(word, self.word2id["<unk>"])

    def words2indices(self, sents: List[List[str]]) -> List[List[int]]:
        return [[self[token] for token in sent] for sent in sents]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        word_ids = self.words2indices(sents)
        sents_t = pad_textual_data(word_ids, self["<pad>"])
        return torch.tensor(sents_t, dtype=torch.long, device=device)


def _load_word2id(vocab_path: Path) -> Dict[str, int]:
    raw = json.loads(vocab_path.read_text())
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    if isinstance(raw, list):
        return {str(word): idx for idx, word in enumerate(raw)}
    raise ValueError("vocab.json must be a dict[word->id] or list[word].")


def load_text_artifact(artifact_uri: str) -> TextArtifact:
    root = Path(resolve_uri_to_path(artifact_uri))
    if not root.exists():
        raise FileNotFoundError(f"text artifact path not found: {root}")
    if not root.is_dir():
        raise ValueError("artifact_uri must point to a directory with vocab.json, embeddings.npy, metadata.json")

    vocab_path = root / "vocab.json"
    embeddings_path = root / "embeddings.npy"
    if not vocab_path.exists():
        raise FileNotFoundError(f"missing vocab file: {vocab_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"missing embeddings file: {embeddings_path}")

    word2id = _load_word2id(vocab_path)
    embedding_matrix = np.load(embeddings_path)
    if embedding_matrix.ndim != 2:
        raise ValueError("embeddings.npy must be a rank-2 matrix [vocab_size, embed_dim].")

    max_id = max(word2id.values()) if word2id else -1
    if max_id >= embedding_matrix.shape[0]:
        raise ValueError("embedding matrix row count is smaller than max vocab id.")

    return TextArtifact(
        word2id=word2id,
        embedding_matrix=embedding_matrix,
    )
