import json
import os
from pathlib import Path
import sys
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = next((parent for parent in THIS_FILE.parents if (parent / "service_registry.py").exists()), THIS_FILE.parents[2])
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from service_registry import build_service_lifespan
from .redis_cache import (
    RedisTextCache,
    build_prediction_cache_key,
    build_prediction_lock_key,
)
from tall_text import (
    load_text_artifact,
    normalize_lookup_text,
    resolve_sentence_embedding,
    resolve_uri_to_path,
)


class ServiceModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class JobEnvelope(ServiceModel):
    job_id: str
    trace_id: str
    dataset: str
    created_at: str
    version: str = "v1"


class TextBatchRequest(ServiceModel):
    sentences: list[list[str]]
    artifact_uri: str
    expected_representation: str | None = None


class TextBatchResponse(ServiceModel):
    sentence_embeddings: list[list[float]]
    embedding_shape: list[int]
    representation_type: str


class ProcessTextFileRequest(ServiceModel):
    job: JobEnvelope | None = None
    input_text_uri: str
    artifact_uri: str
    start_time: float | None = None
    end_time: float | None = None
    base_name: str | None = None
    video_features_uri: str | None = None
    downstream_mode: Literal["inference", "training"] | None = None
    downstream_service_url: str | None = None
    model_uri: str | None = None
    output_model_uri: str | None = None
    top_n: int = 5
    hyperparams: dict | None = None
    artifact_lookup_key: str | None = None
    expected_representation: str | None = None


class ProcessTextFileResponse(ServiceModel):
    status: str
    metadata: dict
    downstream: dict


class ProcessAlignedTextRequest(ServiceModel):
    job: JobEnvelope | None = None
    input_alignment_uri: str
    artifact_uri: str
    video_features_uri: str | None = None
    fps: float = Field(default=30.0, gt=0.0)
    row_indices: list[int] | None = None
    base_name_prefix: str | None = None
    output_split_uri: str | None = None
    expected_representation: str | None = None


class ProcessAlignedTextResponse(ServiceModel):
    status: str
    metadata: dict
    records: list[dict]


def derive_base_name(input_uri: str) -> str:
    name = Path(resolve_uri_to_path(input_uri)).name
    for suffix in [".aligned.tsv", ".txt", ".tp.json", ".vf.pt", ".vp.npy", ".mp4", ".avi"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def build_text_processed_uri(base_name: str) -> str:
    return f"shared://text/processed/{base_name}.tp.json"


def build_visual_feature_uri(base_name: str) -> str:
    return f"shared://features/visual/{base_name}.vf.pt"


def build_dispatch_uri(base_name: str, downstream_mode: str) -> str:
    return f"shared://results/dispatch/{base_name}.{downstream_mode}.json"


def resolve_base_name(explicit_base_name: str | None, fallback_uri: str) -> str:
    return explicit_base_name or derive_base_name(fallback_uri)


def resolve_video_features_uri(explicit_uri: str | None, base_name: str) -> str:
    return explicit_uri or build_visual_feature_uri(base_name)


def persist_text_processed_record(base_name: str, record: dict) -> str:
    text_processed_uri = build_text_processed_uri(base_name)
    text_processed_path = resolve_uri_to_path(text_processed_uri)
    os.makedirs(os.path.dirname(text_processed_path) or ".", exist_ok=True)

    with open(text_processed_path, "w") as f:
        json.dump(record, f)

    return text_processed_uri


def maybe_trigger_downstream(
    base_name: str,
    text_processed_uri: str,
    video_features_uri: str,
    downstream_mode: str | None,
    downstream_service_url: str | None,
    model_uri: str | None,
    output_model_uri: str | None,
    top_n: int,
    hyperparams: dict | None,
) -> dict:
    if downstream_mode is None or downstream_service_url is None:
        return {"triggered": False, "reason": "downstream_not_configured"}

    visual_feature_path = resolve_uri_to_path(video_features_uri)
    if not os.path.exists(visual_feature_path):
        return {
            "triggered": False,
            "reason": "visual_feature_not_ready",
            "expected_visual_feature_uri": video_features_uri,
        }

    if downstream_mode == "inference" and model_uri is None:
        return {"triggered": False, "reason": "missing_model_uri"}
    if downstream_mode == "training" and output_model_uri is None:
        return {"triggered": False, "reason": "missing_output_model_uri"}

    dispatch_uri = build_dispatch_uri(base_name, downstream_mode)
    dispatch_path = resolve_uri_to_path(dispatch_uri)
    os.makedirs(os.path.dirname(dispatch_path) or ".", exist_ok=True)

    try:
        with open(dispatch_path, "x") as f:
            json.dump({"status": "claimed", "base_name": base_name, "mode": downstream_mode}, f)
    except FileExistsError:
        return {"triggered": False, "reason": "already_dispatched", "dispatch_uri": dispatch_uri}

    if downstream_mode == "inference":
        payload = {
            "model_uri": model_uri,
            "video_features_uri": video_features_uri,
            "text_processed_uri": text_processed_uri,
            "top_n": top_n,
        }
        endpoint = "/infer/ground"
    elif downstream_mode == "training":
        payload = {
            "video_features_uri": video_features_uri,
            "text_processed_uri": text_processed_uri,
            "output_model_uri": output_model_uri,
            "hyperparams": hyperparams or {},
        }
        endpoint = "/jobs/train-from-artifacts"
    else:
        return {"triggered": False, "reason": f"unsupported_mode:{downstream_mode}"}

    response = requests.post(f"{downstream_service_url.rstrip('/')}{endpoint}", json=payload, timeout=30)
    response.raise_for_status()
    body = response.json()

    with open(dispatch_path, "w") as f:
        json.dump({"status": "completed", "endpoint": endpoint, "response": body}, f)

    return {
        "triggered": True,
        "dispatch_uri": dispatch_uri,
        "downstream_response": body,
    }


def build_query_text(action: str, agent: str, object_name: str, location: str) -> str:
    parts = []
    if action:
        parts.append(action)
    if object_name:
        parts.append(object_name)
    if location:
        parts.append(f"from {location}")
    if agent:
        parts.append(f"using {agent}")
    return " ".join(parts)


def parse_alignment_rows(input_alignment_uri: str) -> list[dict]:
    input_alignment_path = resolve_uri_to_path(input_alignment_uri)
    rows = []

    with open(input_alignment_path) as f:
        for row_index, raw_line in enumerate(f):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            cols = line.split("\t")
            if len(cols) < 2:
                raise ValueError(f"alignment row {row_index} must include at least start and end frame")

            start_frame = int(cols[0])
            end_frame = int(cols[1])
            action = cols[2].strip() if len(cols) > 2 else ""
            agent = cols[3].strip() if len(cols) > 3 else ""
            object_name = cols[4].strip() if len(cols) > 4 else ""
            location = cols[5].strip() if len(cols) > 5 else ""

            rows.append(
                {
                    "row_index": row_index,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "action": action,
                    "agent": agent,
                    "object": object_name,
                    "location": location,
                    "query_text": build_query_text(action, agent, object_name, location),
                }
            )

    if not rows:
        raise ValueError(f"alignment file contains no rows: {input_alignment_path}")

    return rows


def select_alignment_rows(rows: list[dict], row_indices: list[int] | None) -> list[dict]:
    if row_indices is None:
        return rows

    selected = []
    for row_index in row_indices:
        if row_index < 0 or row_index >= len(rows):
            raise ValueError(f"row index out of range: {row_index}")
        selected.append(rows[row_index])
    return selected


def ensure_supported_representation(expected_representation: str | None) -> None:
    if expected_representation is None:
        return
    if expected_representation not in {"sentence_embedding", "skip_thought_lookup"}:
        raise HTTPException(
            status_code=400,
            detail="TALL text processing supports sentence_embedding representations",
        )


app = FastAPI(
    title="Text Processing Service",
    version="0.1.0",
    lifespan=build_service_lifespan(
        logical_service="text-processing-service",
        default_port=8003,
        service_version="0.1.0",
        api_title="Text Processing Service",
    ),
)
TEXT_CACHE = RedisTextCache()


@app.get("/registry/self")
def registry_self() -> dict:
    return app.state.service_registry


def resolve_cached_sentence_embedding(
    *,
    artifact,
    raw_text: str,
    base_name: str,
    artifact_lookup_key: str | None,
) -> tuple[dict, dict]:
    normalized_text = normalize_lookup_text(raw_text)
    cache_key = build_prediction_cache_key(
        artifact_uri=artifact.artifact_uri,
        normalized_text=normalized_text,
        base_name=base_name,
        lookup_hint=artifact_lookup_key,
    )

    if not TEXT_CACHE.enabled:
        sentence_embedding, lookup_key = resolve_sentence_embedding(
            artifact,
            raw_text,
            base_name=base_name,
            lookup_hint=artifact_lookup_key,
        )
        return (
            {
                "sentence_embedding": sentence_embedding.tolist(),
                "sentence_embedding_dim": int(sentence_embedding.shape[0]),
                "sentence_embedding_lookup_key": lookup_key,
                "normalized_text": normalized_text,
                "artifact_representation_type": artifact.representation_type,
            },
            {
                "cache_enabled": False,
                "cache_hit": False,
                "cache_key": cache_key,
            },
        )

    cached = TEXT_CACHE.get_json(cache_key)
    if cached is not None:
        sentence_embedding = cached.get("sentence_embedding")
        lookup_key = cached.get("sentence_embedding_lookup_key")
        sentence_embedding_dim = cached.get("sentence_embedding_dim")
        cached_normalized_text = cached.get("normalized_text")
        if (
            isinstance(sentence_embedding, list)
            and isinstance(lookup_key, str)
            and isinstance(sentence_embedding_dim, int)
            and isinstance(cached_normalized_text, str)
        ):
            return (
                {
                    "sentence_embedding": sentence_embedding,
                    "sentence_embedding_dim": sentence_embedding_dim,
                    "sentence_embedding_lookup_key": lookup_key,
                    "normalized_text": cached_normalized_text,
                    "artifact_representation_type": str(
                        cached.get("artifact_representation_type") or artifact.representation_type
                    ),
                },
                {
                    "cache_enabled": True,
                    "cache_hit": True,
                    "cache_key": cache_key,
                },
            )

    lock_key = build_prediction_lock_key(cache_key)
    lock_acquired = TEXT_CACHE.acquire_lock(lock_key)
    try:
        if lock_acquired:
            cached = TEXT_CACHE.get_json(cache_key)
            if cached is not None:
                sentence_embedding = cached.get("sentence_embedding")
                lookup_key = cached.get("sentence_embedding_lookup_key")
                sentence_embedding_dim = cached.get("sentence_embedding_dim")
                cached_normalized_text = cached.get("normalized_text")
                if (
                    isinstance(sentence_embedding, list)
                    and isinstance(lookup_key, str)
                    and isinstance(sentence_embedding_dim, int)
                    and isinstance(cached_normalized_text, str)
                ):
                    return (
                        {
                            "sentence_embedding": sentence_embedding,
                            "sentence_embedding_dim": sentence_embedding_dim,
                            "sentence_embedding_lookup_key": lookup_key,
                            "normalized_text": cached_normalized_text,
                            "artifact_representation_type": str(
                                cached.get("artifact_representation_type") or artifact.representation_type
                            ),
                        },
                        {
                            "cache_enabled": True,
                            "cache_hit": True,
                            "cache_key": cache_key,
                        },
                    )

        sentence_embedding, lookup_key = resolve_sentence_embedding(
            artifact,
            raw_text,
            base_name=base_name,
            lookup_hint=artifact_lookup_key,
        )
        resolved = {
            "sentence_embedding": sentence_embedding.tolist(),
            "sentence_embedding_dim": int(sentence_embedding.shape[0]),
            "sentence_embedding_lookup_key": lookup_key,
            "normalized_text": normalized_text,
            "artifact_representation_type": artifact.representation_type,
        }
        TEXT_CACHE.set_json(cache_key, resolved)
        return (
            resolved,
            {
                "cache_enabled": True,
                "cache_hit": False,
                "cache_key": cache_key,
            },
        )
    finally:
        if lock_acquired:
            TEXT_CACHE.release_lock(lock_key)


@app.post("/text/batch", response_model=TextBatchResponse)
def process_text_batch(payload: TextBatchRequest) -> TextBatchResponse:
    ensure_supported_representation(payload.expected_representation)
    artifact = load_text_artifact(artifact_uri=payload.artifact_uri)
    sentence_embeddings = []

    for tokens in payload.sentences:
        query_text = " ".join(token.strip() for token in tokens if token.strip())
        sentence_embedding, _ = resolve_sentence_embedding(artifact, query_text)
        sentence_embeddings.append(sentence_embedding.tolist())

    embedding_dim = artifact.embedding_dim if sentence_embeddings else 0
    return TextBatchResponse(
        sentence_embeddings=sentence_embeddings,
        embedding_shape=[len(sentence_embeddings), embedding_dim],
        representation_type=artifact.representation_type,
    )


@app.post("/jobs/process-text", response_model=ProcessTextFileResponse)
def process_text_file(payload: ProcessTextFileRequest) -> ProcessTextFileResponse:
    ensure_supported_representation(payload.expected_representation)
    artifact = load_text_artifact(artifact_uri=payload.artifact_uri)

    base_name = resolve_base_name(payload.base_name, payload.input_text_uri)
    video_features_uri = resolve_video_features_uri(payload.video_features_uri, derive_base_name(payload.input_text_uri))
    input_text_path = resolve_uri_to_path(payload.input_text_uri)

    with open(input_text_path) as f:
        raw_text = f.read()

    cached_record, cache_metadata = resolve_cached_sentence_embedding(
        artifact=artifact,
        raw_text=raw_text,
        base_name=base_name,
        artifact_lookup_key=payload.artifact_lookup_key,
    )

    record = {
        "base_name": base_name,
        "source_text_uri": payload.input_text_uri,
        "artifact_uri": payload.artifact_uri,
        "video_features_uri": video_features_uri,
        "query_text": raw_text,
        "start_time": payload.start_time,
        "end_time": payload.end_time,
    }
    record.update(cached_record)
    text_processed_uri = persist_text_processed_record(base_name, record)

    metadata = {
        "base_name": base_name,
        "text_processed_uri": text_processed_uri,
        "video_features_uri": video_features_uri,
        "sentence_embedding_dim": cached_record["sentence_embedding_dim"],
        "sentence_embedding_lookup_key": cached_record["sentence_embedding_lookup_key"],
        "artifact_representation_type": cached_record["artifact_representation_type"],
        "start_time": payload.start_time,
        "end_time": payload.end_time,
        "normalized_text": cached_record["normalized_text"],
        "cache_enabled": cache_metadata["cache_enabled"],
        "cache_hit": cache_metadata["cache_hit"],
        "cache_key": cache_metadata["cache_key"],
    }
    downstream = maybe_trigger_downstream(
        base_name=base_name,
        text_processed_uri=text_processed_uri,
        video_features_uri=video_features_uri,
        downstream_mode=payload.downstream_mode,
        downstream_service_url=payload.downstream_service_url,
        model_uri=payload.model_uri,
        output_model_uri=payload.output_model_uri,
        top_n=payload.top_n,
        hyperparams=payload.hyperparams,
    )

    return ProcessTextFileResponse(status="accepted", metadata=metadata, downstream=downstream)


@app.post("/jobs/process-aligned-text", response_model=ProcessAlignedTextResponse)
def process_aligned_text(payload: ProcessAlignedTextRequest) -> ProcessAlignedTextResponse:
    ensure_supported_representation(payload.expected_representation)
    artifact = load_text_artifact(artifact_uri=payload.artifact_uri)

    video_base_name = derive_base_name(payload.input_alignment_uri)
    video_features_uri = resolve_video_features_uri(payload.video_features_uri, video_base_name)
    rows = select_alignment_rows(parse_alignment_rows(payload.input_alignment_uri), payload.row_indices)

    records = []
    split_records = []
    prefix = payload.base_name_prefix or video_base_name

    for row in rows:
        base_name = f"{prefix}-e{row['row_index'] + 1}"
        sentence_embedding, lookup_key = resolve_sentence_embedding(
            artifact,
            row["query_text"],
            base_name=video_base_name,
            row_index=row["row_index"],
            lookup_hint=base_name,
        )
        start_time = row["start_frame"] / payload.fps
        end_time = row["end_frame"] / payload.fps

        record = {
            "base_name": base_name,
            "source_alignment_uri": payload.input_alignment_uri,
            "row_index": row["row_index"],
            "artifact_uri": payload.artifact_uri,
            "artifact_representation_type": artifact.representation_type,
            "video_features_uri": video_features_uri,
            "query_text": row["query_text"],
            "normalized_text": normalize_lookup_text(row["query_text"]),
            "action": row["action"],
            "agent": row["agent"],
            "object": row["object"],
            "location": row["location"],
            "sentence_embedding": sentence_embedding.tolist(),
            "sentence_embedding_dim": int(sentence_embedding.shape[0]),
            "sentence_embedding_lookup_key": lookup_key,
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "start_time": start_time,
            "end_time": end_time,
        }
        text_processed_uri = persist_text_processed_record(base_name, record)

        records.append(
            {
                "base_name": base_name,
                "text_processed_uri": text_processed_uri,
                "video_features_uri": video_features_uri,
                "row_index": row["row_index"],
                "query_text": row["query_text"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "start_time": start_time,
                "end_time": end_time,
                "sentence_embedding_dim": int(sentence_embedding.shape[0]),
                "sentence_embedding_lookup_key": lookup_key,
            }
        )
        split_records.append(
            {
                "base_name": base_name,
                "video_features_uri": video_features_uri,
                "text_processed_uri": text_processed_uri,
            }
        )

    split_uri = payload.output_split_uri
    if split_uri is not None:
        split_path = resolve_uri_to_path(split_uri)
        os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)
        with open(split_path, "w") as f:
            json.dump(split_records, f, indent=2)
            f.write("\n")

    metadata = {
        "source_alignment_uri": payload.input_alignment_uri,
        "video_features_uri": video_features_uri,
        "rows_processed": len(records),
        "fps": payload.fps,
        "output_split_uri": split_uri,
        "artifact_representation_type": artifact.representation_type,
        "sentence_embedding_dim": artifact.embedding_dim,
    }
    return ProcessAlignedTextResponse(status="accepted", metadata=metadata, records=records)
