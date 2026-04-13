from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch


def resolve_uri_to_path(uri: str) -> str:
    storage_root = Path(
        os.environ.get(
            "IMPLEMENTATION_STORAGE_ROOT",
            Path(__file__).resolve().parent / "storage",
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


def normalize_lookup_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _load_lookup_index(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"sentence lookup file must be a JSON object: {path}")

    lookup_index: dict[str, int] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            if "index" not in value:
                raise ValueError(f"sentence lookup entry is missing 'index': {path} -> {key}")
            value = value["index"]
        lookup_index[str(key)] = int(value)
    return lookup_index


@dataclass(frozen=True)
class TextArtifact:
    artifact_uri: str
    representation_type: str
    embedding_matrix: np.ndarray
    lookup_index: dict[str, int]
    metadata: dict[str, Any]

    @property
    def embedding_dim(self) -> int:
        if self.embedding_matrix.ndim != 2:
            raise ValueError("embedding matrix must be rank-2")
        return int(self.embedding_matrix.shape[1])


@lru_cache(maxsize=16)
def load_text_artifact(artifact_uri: str) -> TextArtifact:
    root = Path(resolve_uri_to_path(artifact_uri))
    if not root.exists():
        raise FileNotFoundError(f"text artifact path not found: {root}")
    if not root.is_dir():
        raise ValueError("artifact_uri must point to a directory")

    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        legacy_vocab = root / "vocab.json"
        legacy_embeddings = root / "embeddings.npy"
        if legacy_vocab.exists() and legacy_embeddings.exists():
            raise ValueError(
                "legacy token-embedding artifact detected. "
                "TALL now expects precomputed skip-thought-style sentence embeddings "
                "with metadata.json, sentence_lookup.json, and sentence_embeddings.npy."
            )
        raise FileNotFoundError(f"missing metadata file: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata.json must contain a JSON object: {metadata_path}")

    representation_type = str(metadata.get("representation_type") or "").strip()
    if representation_type not in {"skip_thought_lookup", "sentence_lookup"}:
        raise ValueError(
            "unsupported text artifact representation_type. "
            f"Expected 'skip_thought_lookup' or 'sentence_lookup', got {representation_type!r}."
        )

    embeddings_name = str(metadata.get("embeddings_file") or "sentence_embeddings.npy")
    lookup_name = str(metadata.get("lookup_file") or "sentence_lookup.json")
    embeddings_path = root / embeddings_name
    lookup_path = root / lookup_name

    if not embeddings_path.exists():
        raise FileNotFoundError(f"missing sentence embeddings file: {embeddings_path}")
    if not lookup_path.exists():
        raise FileNotFoundError(f"missing sentence lookup file: {lookup_path}")

    embedding_matrix = np.load(embeddings_path)
    if embedding_matrix.ndim != 2:
        raise ValueError("sentence embeddings must be rank-2 [num_sentences, embedding_dim]")

    lookup_index = _load_lookup_index(lookup_path)
    if lookup_index:
        max_id = max(lookup_index.values())
        if max_id >= embedding_matrix.shape[0]:
            raise ValueError("sentence lookup references an embedding row that does not exist")

    expected_dim = metadata.get("embedding_dim")
    if expected_dim is not None and int(expected_dim) != int(embedding_matrix.shape[1]):
        raise ValueError(
            "metadata embedding_dim does not match sentence embedding matrix width: "
            f"{expected_dim} != {embedding_matrix.shape[1]}"
        )

    return TextArtifact(
        artifact_uri=artifact_uri,
        representation_type=representation_type,
        embedding_matrix=embedding_matrix.astype(np.float32, copy=False),
        lookup_index=lookup_index,
        metadata=metadata,
    )


def build_lookup_candidates(
    raw_text: str,
    *,
    base_name: str | None = None,
    row_index: int | None = None,
    lookup_hint: str | None = None,
) -> list[str]:
    candidates: list[str] = []

    def add(candidate: str | None) -> None:
        if candidate is None:
            return
        normalized = str(candidate).strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    add(lookup_hint)
    if base_name is not None and row_index is not None:
        add(f"{base_name}#row:{row_index}")
        add(f"{base_name}:row:{row_index}")
        add(f"{base_name}#{row_index}")
        add(f"{base_name}-e{row_index + 1}")
    add(base_name)
    add(raw_text.strip())
    add(normalize_lookup_text(raw_text))
    return candidates


def resolve_sentence_embedding(
    artifact: TextArtifact,
    raw_text: str,
    *,
    base_name: str | None = None,
    row_index: int | None = None,
    lookup_hint: str | None = None,
) -> tuple[np.ndarray, str]:
    for lookup_key in build_lookup_candidates(
        raw_text,
        base_name=base_name,
        row_index=row_index,
        lookup_hint=lookup_hint,
    ):
        index = artifact.lookup_index.get(lookup_key)
        if index is None:
            continue
        vector = artifact.embedding_matrix[int(index)]
        if vector.ndim != 1:
            raise ValueError(f"sentence embedding lookup must resolve to a rank-1 vector: {lookup_key}")
        return vector.astype(np.float32, copy=True), lookup_key

    raise KeyError(
        "no sentence embedding found for query text in text artifact. "
        f"Tried keys: {build_lookup_candidates(raw_text, base_name=base_name, row_index=row_index, lookup_hint=lookup_hint)}"
    )


def sentence_embedding_from_text_processed(text_processed: dict) -> torch.Tensor:
    sentence_embedding = text_processed.get("sentence_embedding")
    if not isinstance(sentence_embedding, list):
        raise ValueError("text processed file must contain 'sentence_embedding' as a list")

    array = np.asarray(sentence_embedding, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("sentence_embedding must be rank-1")
    return torch.from_numpy(array)
