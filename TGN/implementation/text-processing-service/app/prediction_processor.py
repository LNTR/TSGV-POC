from dataclasses import dataclass, field
import os
import re
from typing import Protocol

import numpy as np
import torch

from .redis_cache import (
    RedisTextCache,
    build_prediction_cache_key,
    build_prediction_lock_key,
    build_prediction_semantic_index_key,
    normalize_prediction_text,
)
from .vocab import Vocab


DEFAULT_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
    }
)


def tokenize_text(text: str) -> list[str]:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    return [token for token in tokens if token.strip()]


def semantic_similarity_enabled() -> bool:
    return os.environ.get("TEXT_PREDICTION_SEMANTIC_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def semantic_similarity_threshold() -> float:
    raw = os.environ.get("TEXT_PREDICTION_SEMANTIC_THRESHOLD", "0.90").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.90
    return min(max(value, -1.0), 1.0)


def semantic_similarity_max_candidates() -> int:
    raw = os.environ.get("TEXT_PREDICTION_SEMANTIC_MAX_CANDIDATES", "256").strip()
    try:
        value = int(raw)
    except ValueError:
        return 256
    return max(value, 1)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float | None:
    if len(vec_a) != len(vec_b) or not vec_a:
        return None

    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return None
    return float(np.dot(a, b) / denom)


@dataclass
class PredictionProcessingContext:
    raw_text: str
    artifact_uri: str
    vocab: Vocab
    embedding_matrix: np.ndarray
    normalized_text: str = ""
    semantic_tokens: list[str] = field(default_factory=list)
    semantic_normalized_text: str = ""
    token_ids: list[int] = field(default_factory=list)
    length_t: int = 0
    cache_hit: bool = False
    cache_key: str | None = None
    cache_enabled: bool = False
    semantic_enabled: bool = False
    semantic_vector_ready: bool = False
    semantic_known_token_count: int = 0
    semantic_match: dict | None = None
    semantic_vector: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.normalized_text:
            self.normalized_text = normalize_prediction_text(self.raw_text)


class PredictionProcessor(Protocol):
    def process(self, context: PredictionProcessingContext) -> PredictionProcessingContext:
        ...


class BasePredictionProcessor:
    def process(self, context: PredictionProcessingContext) -> PredictionProcessingContext:
        tokens = tokenize_text(context.raw_text)
        tensor = context.vocab.to_input_tensor([tokens], torch.device("cpu"))
        context.token_ids = tensor[0].tolist()
        context.length_t = len(tokens)
        return context


class PredictionProcessorDecorator:
    def __init__(self, wrapped: PredictionProcessor) -> None:
        self.wrapped = wrapped

    def process(self, context: PredictionProcessingContext) -> PredictionProcessingContext:
        return self.wrapped.process(context)


class StopWordNormalizationDecorator(PredictionProcessorDecorator):
    def __init__(self, wrapped: PredictionProcessor, stop_words: set[str] | None = None) -> None:
        super().__init__(wrapped)
        self.stop_words = set(stop_words or DEFAULT_STOP_WORDS)

    def process(self, context: PredictionProcessingContext) -> PredictionProcessingContext:
        semantic_tokens = [token for token in tokenize_text(context.normalized_text) if token not in self.stop_words]
        if not semantic_tokens:
            semantic_tokens = tokenize_text(context.normalized_text)

        context.semantic_tokens = semantic_tokens
        context.semantic_normalized_text = " ".join(semantic_tokens)
        return self.wrapped.process(context)


class SemanticMatchDecorator(PredictionProcessorDecorator):
    def __init__(
        self,
        wrapped: PredictionProcessor,
        *,
        cache: RedisTextCache,
        enabled: bool,
        threshold: float,
        max_candidates: int,
    ) -> None:
        super().__init__(wrapped)
        self.cache = cache
        self.enabled = enabled
        self.threshold = threshold
        self.max_candidates = max_candidates

    def process(self, context: PredictionProcessingContext) -> PredictionProcessingContext:
        context = self.wrapped.process(context)
        context.semantic_enabled = self.enabled

        if context.cache_hit or not self.enabled or not self.cache.enabled:
            return context

        semantic_vector, known_token_count = self._build_semantic_vector(context)
        context.semantic_vector = semantic_vector
        context.semantic_vector_ready = semantic_vector is not None
        context.semantic_known_token_count = known_token_count
        if semantic_vector is None:
            return context

        index_key = build_prediction_semantic_index_key(context.artifact_uri)
        best_match = None
        best_score = self.threshold

        for candidate_key in self.cache.get_index_members(index_key, limit=self.max_candidates):
            if candidate_key == context.cache_key:
                continue

            candidate = self.cache.get_json(candidate_key)
            if candidate is None:
                continue

            candidate_vector = candidate.get("semantic_vector")
            candidate_text = candidate.get("normalized_text")
            if not isinstance(candidate_vector, list) or not isinstance(candidate_text, str):
                continue
            if candidate_text == context.normalized_text:
                continue

            score = cosine_similarity(semantic_vector, candidate_vector)
            if score is None or score < best_score:
                continue

            best_score = score
            best_match = {
                "cache_key": candidate_key,
                "normalized_text": candidate_text,
                "score": round(score, 6),
                "length_t": candidate.get("length_t"),
                "token_ids": candidate.get("token_ids"),
            }

        context.semantic_match = best_match
        return context

    def _build_semantic_vector(self, context: PredictionProcessingContext) -> tuple[list[float] | None, int]:
        unk_id = int(context.vocab.word2id.get("<unk>", 1))
        semantic_ids = [context.vocab[token] for token in context.semantic_tokens]
        known_ids = [
            token_id
            for token_id in semantic_ids
            if token_id != unk_id and 0 <= token_id < context.embedding_matrix.shape[0]
        ]
        if not known_ids:
            return None, 0

        vec = context.embedding_matrix[np.array(known_ids, dtype=np.int64)].mean(axis=0)
        norm = float(np.linalg.norm(vec))
        if norm <= 0.0:
            return None, len(known_ids)

        return (vec / norm).astype(np.float32).tolist(), len(known_ids)


class ExactCacheDecorator(PredictionProcessorDecorator):
    def __init__(self, wrapped: PredictionProcessor, *, cache: RedisTextCache, semantic_enabled: bool) -> None:
        super().__init__(wrapped)
        self.cache = cache
        self.semantic_enabled = semantic_enabled

    def process(self, context: PredictionProcessingContext) -> PredictionProcessingContext:
        context.cache_enabled = self.cache.enabled and bool(context.normalized_text)
        context.semantic_enabled = self.semantic_enabled
        context.cache_key = build_prediction_cache_key(
            artifact_uri=context.artifact_uri,
            normalized_text=context.normalized_text,
        )

        if context.cache_enabled:
            cached = self.cache.get_json(context.cache_key)
            if self._apply_cached_payload(context, cached):
                return context

        lock_key = build_prediction_lock_key(context.cache_key)
        lock_acquired = context.cache_enabled and self.cache.acquire_lock(lock_key)
        try:
            if lock_acquired:
                cached = self.cache.get_json(context.cache_key)
                if self._apply_cached_payload(context, cached):
                    return context

            context = self.wrapped.process(context)
            if context.cache_enabled:
                cached_value = {
                    "normalized_text": context.normalized_text,
                    "semantic_normalized_text": context.semantic_normalized_text,
                    "artifact_uri": context.artifact_uri,
                    "token_ids": context.token_ids,
                    "length_t": context.length_t,
                }
                if context.semantic_vector is not None:
                    cached_value["semantic_vector"] = context.semantic_vector

                self.cache.set_json(context.cache_key, cached_value)
                if context.semantic_vector is not None:
                    self.cache.add_index_member(
                        build_prediction_semantic_index_key(context.artifact_uri),
                        context.cache_key,
                    )

            return context
        finally:
            if lock_acquired:
                self.cache.release_lock(lock_key)

    def _apply_cached_payload(self, context: PredictionProcessingContext, cached: dict | None) -> bool:
        if cached is None:
            return False

        token_ids = cached.get("token_ids")
        length_t = cached.get("length_t")
        if not isinstance(token_ids, list) or not isinstance(length_t, int):
            return False

        context.token_ids = [int(token_id) for token_id in token_ids]
        context.length_t = int(length_t)
        context.cache_hit = True
        context.semantic_vector_ready = isinstance(cached.get("semantic_vector"), list)
        context.semantic_normalized_text = str(cached.get("semantic_normalized_text") or "")
        context.semantic_match = None
        return True


def build_prediction_processor(cache: RedisTextCache) -> PredictionProcessor:
    semantic_enabled = semantic_similarity_enabled()
    processor: PredictionProcessor = BasePredictionProcessor()
    processor = StopWordNormalizationDecorator(processor)
    processor = SemanticMatchDecorator(
        processor,
        cache=cache,
        enabled=semantic_enabled,
        threshold=semantic_similarity_threshold(),
        max_candidates=semantic_similarity_max_candidates(),
    )
    processor = ExactCacheDecorator(processor, cache=cache, semantic_enabled=semantic_enabled)
    return processor
