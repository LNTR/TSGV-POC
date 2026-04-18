import hashlib
import json
import os
import math
from pathlib import Path
import shutil
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from skimage import transform


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


def derive_base_name(input_uri: str) -> str:
    return Path(resolve_uri_to_path(input_uri)).stem


def build_visual_preprocess_uri(input_video_uri: str) -> str:
    return f"shared://frames/processed/{derive_base_name(input_video_uri)}.vp.npy"


def build_visual_feature_uri(input_video_uri: str) -> str:
    return f"shared://features/visual/{derive_base_name(input_video_uri)}.vf.pt"


def build_visual_preprocess_metadata_uri(input_video_uri: str) -> str:
    return f"shared://frames/metadata/{derive_base_name(input_video_uri)}.vp.meta.json"


def load_json_if_exists(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def compute_file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_preprocess_cache_key(video_hash: str, output_frame_size: int, sample_every_sec: int) -> str:
    material = json.dumps(
        {
            "video_hash": video_hash,
            "output_frame_size": int(output_frame_size),
            "sample_every_sec": int(sample_every_sec),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"visual:preprocess:v1:{digest}"


def get_redis_client():
    redis_url = os.environ.get("REDIS_URL", "").strip()
    if not redis_url:
        return None
    try:
        import redis
    except ImportError:
        return None
    return redis.Redis.from_url(redis_url, decode_responses=True)


def get_cached_payload(cache_key: str) -> dict | None:
    client = get_redis_client()
    if client is None:
        return None
    try:
        raw = client.get(cache_key)
    except Exception:
        return None
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def set_cached_payload(cache_key: str, payload: dict) -> None:
    client = get_redis_client()
    if client is None:
        return
    try:
        client.set(cache_key, json.dumps(payload), ex=int(os.environ.get("VISUAL_PREPROCESS_CACHE_TTL_SEC", "604800")))
    except Exception:
        return


def same_path(path_a: str, path_b: str) -> bool:
    return str(Path(path_a).resolve()) == str(Path(path_b).resolve())


def copy_file_if_needed(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if same_path(src, dst):
        return
    shutil.copy2(src, dst)


def extract_sampled_frames(
    input_video_path: str,
    output_frame_size: tuple[int, int],
    sample_every_sec: int,
) -> np.ndarray:
    cap = cv2.VideoCapture(input_video_path)
    success = True
    frames = []

    current_frame = 0
    fps = max(1, math.ceil(cap.get(cv2.CAP_PROP_FPS)))
    sample_every_n_frames = fps * sample_every_sec

    while success:
        success, frame = cap.read()
        if not success:
            break

        if current_frame % sample_every_n_frames == 0:
            frame = transform.resize(frame, output_frame_size)
            frames.append(np.expand_dims(frame, axis=0))

        current_frame += 1

    cap.release()

    if not frames:
        return np.zeros((0, output_frame_size[0], output_frame_size[1], 3), dtype=np.float32)

    return np.concatenate(frames).astype(np.float32)


def run_preprocess_job(
    input_video_uri: str,
    output_frame_size: int,
    sample_every_sec: int,
) -> dict:
    output_frames_uri = build_visual_preprocess_uri(input_video_uri)
    metadata_uri = build_visual_preprocess_metadata_uri(input_video_uri)
    input_video_path = resolve_uri_to_path(input_video_uri)
    output_frames_path = resolve_uri_to_path(output_frames_uri)
    metadata_path = resolve_uri_to_path(metadata_uri)

    os.makedirs(os.path.dirname(output_frames_path) or ".", exist_ok=True)
    video_hash = compute_file_sha256(input_video_path)
    cache_key = build_preprocess_cache_key(
        video_hash=video_hash,
        output_frame_size=output_frame_size,
        sample_every_sec=sample_every_sec,
    )

    existing_metadata = load_json_if_exists(metadata_path)
    if existing_metadata is not None and os.path.exists(output_frames_path):
        if (
            existing_metadata.get("video_hash") == video_hash
            and int(existing_metadata.get("output_frame_size", -1)) == int(output_frame_size)
            and int(existing_metadata.get("sample_every_sec", -1)) == int(sample_every_sec)
        ):
            return {
                "base_name": derive_base_name(input_video_uri),
                "frames_uri": output_frames_uri,
                "metadata_uri": metadata_uri,
                "num_frames": int(existing_metadata.get("num_frames", 0)),
                "video_hash": video_hash,
                "cache_hit": True,
                "cache_source": "local_metadata",
                "preprocess_cache_key": cache_key,
            }

    cached = get_cached_payload(cache_key)
    if cached is not None:
        cached_frames_uri = cached.get("frames_uri")
        if isinstance(cached_frames_uri, str):
            cached_frames_path = resolve_uri_to_path(cached_frames_uri)
            if os.path.exists(cached_frames_path):
                copy_file_if_needed(cached_frames_path, output_frames_path)
                metadata = {
                    "base_name": derive_base_name(input_video_uri),
                    "input_video_uri": input_video_uri,
                    "frames_uri": output_frames_uri,
                    "metadata_uri": metadata_uri,
                    "num_frames": int(cached.get("num_frames", 0)),
                    "output_frame_size": int(output_frame_size),
                    "sample_every_sec": int(sample_every_sec),
                    "video_hash": video_hash,
                    "preprocess_cache_key": cache_key,
                    "cache_source": "redis",
                }
                write_json(metadata_path, metadata)
                return {
                    **metadata,
                    "cache_hit": True,
                }

    frames = extract_sampled_frames(
        input_video_path=input_video_path,
        output_frame_size=(output_frame_size, output_frame_size),
        sample_every_sec=sample_every_sec,
    )
    np.save(output_frames_path, frames)
    metadata = {
        "base_name": derive_base_name(input_video_uri),
        "input_video_uri": input_video_uri,
        "frames_uri": output_frames_uri,
        "metadata_uri": metadata_uri,
        "num_frames": int(frames.shape[0]),
        "output_frame_size": int(output_frame_size),
        "sample_every_sec": int(sample_every_sec),
        "video_hash": video_hash,
        "preprocess_cache_key": cache_key,
        "cache_source": "computed",
    }
    write_json(metadata_path, metadata)
    set_cached_payload(
        cache_key,
        {
            "frames_uri": output_frames_uri,
            "metadata_uri": metadata_uri,
            "num_frames": int(frames.shape[0]),
            "video_hash": video_hash,
        },
    )

    return {**metadata, "cache_hit": False}


def forward_to_feature_service(
    feature_service_url: str,
    input_video_uri: str,
    output_frame_size: int,
    sample_every_sec: int,
    encoder: str,
    normalize: dict,
    downstream_mode: str | None,
    downstream_service_url: str | None,
    model_uri: str | None,
    output_model_uri: str | None,
    artifact_uri: str | None,
    top_n: int,
    hyperparams: dict | None,
) -> dict:
    frames_uri = build_visual_preprocess_uri(input_video_uri)
    features_uri = build_visual_feature_uri(input_video_uri)
    timeout_sec = int(os.environ.get("FEATURE_SERVICE_TIMEOUT_SEC", "1800"))
    resolved_feature_service_url = feature_service_url.rstrip("/")
    if not resolved_feature_service_url.endswith("/jobs/features"):
        resolved_feature_service_url = f"{resolved_feature_service_url}/jobs/features"
    response = requests.post(
        resolved_feature_service_url,
        json={
            "frames_uri": frames_uri,
            "output_features_uri": features_uri,
            "encoder": encoder,
            "normalize": normalize,
            "downstream_mode": downstream_mode,
            "downstream_service_url": downstream_service_url,
            "model_uri": model_uri,
            "output_model_uri": output_model_uri,
            "artifact_uri": artifact_uri,
            "top_n": top_n,
            "hyperparams": hyperparams,
        },
        timeout=timeout_sec,
    )
    response.raise_for_status()
    return response.json()
