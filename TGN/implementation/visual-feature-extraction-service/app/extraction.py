import os
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from torchvision import transforms

from .cnn_encoder import VGG16


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


def derive_base_name_from_uri(uri: str) -> str:
    name = Path(resolve_uri_to_path(uri)).name
    for suffix in [".vp.npy", ".vf.pt", ".tp.json", ".mp4", ".avi", ".txt", ".npy", ".pt", ".json"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def build_visual_feature_uri(frames_uri: str) -> str:
    return f"shared://features/visual/{derive_base_name_from_uri(frames_uri)}.vf.pt"


def build_text_processed_uri(frames_uri: str) -> str:
    return f"shared://text/processed/{derive_base_name_from_uri(frames_uri)}.tp.json"


def build_frames_metadata_uri(frames_uri: str) -> str:
    return f"shared://frames/metadata/{derive_base_name_from_uri(frames_uri)}.vp.meta.json"


def build_feature_metadata_uri(frames_uri: str) -> str:
    return f"shared://features/metadata/{derive_base_name_from_uri(frames_uri)}.vf.meta.json"


def build_dispatch_uri(base_name: str, downstream_mode: str) -> str:
    return f"shared://results/dispatch/{base_name}.{downstream_mode}.json"


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


def build_feature_cache_key(
    preprocess_cache_key: str,
    encoder: str,
    mean: list[float],
    std: list[float],
) -> str:
    material = json.dumps(
        {
            "preprocess_cache_key": preprocess_cache_key,
            "encoder": encoder,
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"visual:features:v1:{digest}"


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
        client.set(cache_key, json.dumps(payload), ex=int(os.environ.get("VISUAL_FEATURE_CACHE_TTL_SEC", "604800")))
    except Exception:
        return


def same_path(path_a: str, path_b: str) -> bool:
    return str(Path(path_a).resolve()) == str(Path(path_b).resolve())


def copy_file_if_needed(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if same_path(src, dst):
        return
    shutil.copy2(src, dst)


def get_initial_batch_size(num_frames: int) -> int:
    configured = int(os.environ.get("VISUAL_FEATURE_BATCH_SIZE", "4"))
    return max(1, min(configured, num_frames))


def get_allow_cpu_fallback() -> bool:
    return os.environ.get("VISUAL_FEATURE_ALLOW_CPU_FALLBACK", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def encode_frames(
    cnn_encoder: torch.nn.Module,
    frames_tensor: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, str, int]:
    num_frames = int(frames_tensor.shape[0])
    if num_frames == 0:
        return torch.zeros((0, 4096), dtype=torch.float32), device, 0

    preferred_device = device
    batch_size = get_initial_batch_size(num_frames)

    while True:
        try:
            cnn_encoder.to(device)
            outputs = []
            with torch.no_grad():
                for start in range(0, num_frames, batch_size):
                    batch = frames_tensor[start:start + batch_size].to(device, non_blocking=device.startswith("cuda"))
                    outputs.append(cnn_encoder(batch).cpu())
                    del batch
            return torch.cat(outputs, dim=0), device, batch_size
        except torch.OutOfMemoryError:
            if not device.startswith("cuda"):
                raise

            torch.cuda.empty_cache()
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                continue

            if get_allow_cpu_fallback():
                device = "cpu"
                batch_size = get_initial_batch_size(num_frames)
                continue
            raise
        finally:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()


def extract_visual_features(
    frames_uri: str,
    mean: list[float],
    std: list[float],
    encoder: str = "vgg16",
) -> dict:
    if encoder != "vgg16":
        raise ValueError("Only vgg16 is supported in this stub.")

    output_features_uri = build_visual_feature_uri(frames_uri)
    feature_metadata_uri = build_feature_metadata_uri(frames_uri)
    frames_metadata_uri = build_frames_metadata_uri(frames_uri)
    frames_path = resolve_uri_to_path(frames_uri)
    output_features_path = resolve_uri_to_path(output_features_uri)
    feature_metadata_path = resolve_uri_to_path(feature_metadata_uri)
    frames_metadata_path = resolve_uri_to_path(frames_metadata_uri)
    os.makedirs(os.path.dirname(output_features_path) or ".", exist_ok=True)

    frames_metadata = load_json_if_exists(frames_metadata_path)
    preprocess_cache_key = None
    video_hash = None
    if frames_metadata is not None:
        preprocess_cache_key = frames_metadata.get("preprocess_cache_key")
        video_hash = frames_metadata.get("video_hash")

    if not isinstance(preprocess_cache_key, str) or not preprocess_cache_key:
        preprocess_cache_key = f"frames-uri:{frames_uri}"

    feature_cache_key = build_feature_cache_key(
        preprocess_cache_key=preprocess_cache_key,
        encoder=encoder,
        mean=mean,
        std=std,
    )

    existing_metadata = load_json_if_exists(feature_metadata_path)
    if existing_metadata is not None and os.path.exists(output_features_path):
        if existing_metadata.get("feature_cache_key") == feature_cache_key:
            return {
                "base_name": derive_base_name_from_uri(frames_uri),
                "features_uri": output_features_uri,
                "feature_metadata_uri": feature_metadata_uri,
                "frames_metadata_uri": frames_metadata_uri,
                "timesteps": int(existing_metadata.get("timesteps", 0)),
                "feature_dim": int(existing_metadata.get("feature_dim", 4096)),
                "video_hash": video_hash,
                "feature_cache_key": feature_cache_key,
                "cache_hit": True,
                "cache_source": "local_metadata",
            }

    cached = get_cached_payload(feature_cache_key)
    if cached is not None:
        cached_features_uri = cached.get("features_uri")
        if isinstance(cached_features_uri, str):
            cached_features_path = resolve_uri_to_path(cached_features_uri)
            if os.path.exists(cached_features_path):
                copy_file_if_needed(cached_features_path, output_features_path)
                metadata = {
                    "base_name": derive_base_name_from_uri(frames_uri),
                    "features_uri": output_features_uri,
                    "feature_metadata_uri": feature_metadata_uri,
                    "frames_metadata_uri": frames_metadata_uri,
                    "timesteps": int(cached.get("timesteps", 0)),
                    "feature_dim": int(cached.get("feature_dim", 4096)),
                    "video_hash": video_hash,
                    "feature_cache_key": feature_cache_key,
                    "cache_source": "redis",
                }
                write_json(feature_metadata_path, metadata)
                return {**metadata, "cache_hit": True}

    transform_ = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    cnn_encoder = VGG16(pretrained=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    frames = np.load(frames_path)
    if frames.shape[0] == 0:
        empty = torch.zeros((0, 4096), dtype=torch.float32)
        torch.save(empty, output_features_path)
        metadata = {
            "base_name": derive_base_name_from_uri(frames_uri),
            "features_uri": output_features_uri,
            "feature_metadata_uri": feature_metadata_uri,
            "frames_metadata_uri": frames_metadata_uri,
            "timesteps": 0,
            "feature_dim": 4096,
            "video_hash": video_hash,
            "feature_cache_key": feature_cache_key,
            "cache_source": "computed",
        }
        write_json(feature_metadata_path, metadata)
        set_cached_payload(feature_cache_key, metadata)
        return {**metadata, "cache_hit": False}

    frames_tensor = torch.cat([transform_(frame).unsqueeze(dim=0) for frame in frames], dim=0)

    features, actual_device, batch_size_used = encode_frames(
        cnn_encoder=cnn_encoder,
        frames_tensor=frames_tensor,
        device=device,
    )

    torch.save(features, output_features_path)

    metadata = {
        "base_name": derive_base_name_from_uri(frames_uri),
        "features_uri": output_features_uri,
        "feature_metadata_uri": feature_metadata_uri,
        "frames_metadata_uri": frames_metadata_uri,
        "timesteps": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "video_hash": video_hash,
        "feature_cache_key": feature_cache_key,
        "cache_source": "computed",
        "device_used": actual_device,
        "batch_size_used": int(batch_size_used),
    }
    write_json(feature_metadata_path, metadata)
    set_cached_payload(feature_cache_key, metadata)
    return {**metadata, "cache_hit": False}


def maybe_trigger_downstream(
    frames_uri: str,
    features_uri: str,
    downstream_mode: str | None,
    downstream_service_url: str | None,
    model_uri: str | None,
    output_model_uri: str | None,
    artifact_uri: str | None,
    top_n: int,
    hyperparams: dict | None,
) -> dict:
    if downstream_mode is None or downstream_service_url is None:
        return {"triggered": False, "reason": "downstream_not_configured"}

    base_name = derive_base_name_from_uri(frames_uri)
    text_processed_uri = build_text_processed_uri(frames_uri)
    text_processed_path = resolve_uri_to_path(text_processed_uri)
    if not os.path.exists(text_processed_path):
        return {
            "triggered": False,
            "reason": "text_processed_not_ready",
            "expected_text_processed_uri": text_processed_uri,
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
            "video_features_uri": features_uri,
            "text_processed_uri": text_processed_uri,
            "top_n": top_n,
        }
        endpoint = "/infer/ground"
    elif downstream_mode == "training":
        payload = {
            "video_features_uri": features_uri,
            "text_processed_uri": text_processed_uri,
            "output_model_uri": output_model_uri,
            "hyperparams": hyperparams or {},
        }
        endpoint = "/jobs/train-from-artifacts"
    else:
        return {"triggered": False, "reason": f"unsupported_mode:{downstream_mode}"}

    if artifact_uri is not None:
        payload["artifact_uri"] = artifact_uri

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
