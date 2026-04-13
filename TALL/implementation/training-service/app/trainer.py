from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[2]
if str(IMPLEMENTATION_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPLEMENTATION_ROOT))

from tall_model import CTRLConfig, CTRLTemporalLocalizer
from tall_runtime import (
    build_candidate_windows,
    compute_iou,
    decode_predictions,
    load_json,
    load_text_processed,
    load_visual_features,
    resolve_uri_to_path,
    sentence_embedding_from_text_processed,
)


@dataclass
class HyperParams:
    batch_size: int
    lr: float
    semantic_size: int = 1024
    mlp_hidden_size: int = 1000
    max_window_scales: int = 8
    context_size: int = 1
    positive_iou: float = 0.5
    regression_weight: float = 0.01
    sample_every_sec: float = 5.0
    nms_threshold: float = 0.45
    log_every: int = 10
    max_iter: int = 1000
    valid_niter: int = 50
    top_n_eval: int = 5
    patience: int = 2
    max_num_trial: int = 3
    lr_decay: float = 0.5


@dataclass(frozen=True)
class TrainingRecord:
    base_name: str
    video_features_uri: str
    text_processed_uri: str
    artifact_uri: str
    start_time: float
    end_time: float


class TrainingError(RuntimeError):
    pass


def validate_training_text_processed(text_processed_uri: str) -> dict:
    data = load_text_processed(text_processed_uri)
    start_time = data.get("start_time")
    end_time = data.get("end_time")
    if start_time is None or end_time is None:
        path = resolve_uri_to_path(text_processed_uri)
        raise ValueError(
            "training requires 'start_time' and 'end_time' in text processed files: "
            f"{path}"
        )
    return data


def load_training_record(raw: dict, split_name: str, index: int) -> TrainingRecord:
    if not isinstance(raw, dict):
        raise ValueError(f"{split_name} split entry at index {index} must be a JSON object.")

    video_features_uri = raw.get("video_features_uri")
    text_processed_uri = raw.get("text_processed_uri")
    if not video_features_uri:
        raise ValueError(f"{split_name} split entry at index {index} is missing 'video_features_uri'.")
    if not text_processed_uri:
        raise ValueError(f"{split_name} split entry at index {index} is missing 'text_processed_uri'.")

    text_data = validate_training_text_processed(text_processed_uri)
    _ = load_visual_features(video_features_uri)

    base_name = raw.get("base_name") or text_data.get("base_name") or Path(
        resolve_uri_to_path(video_features_uri)
    ).stem
    return TrainingRecord(
        base_name=str(base_name),
        video_features_uri=str(video_features_uri),
        text_processed_uri=str(text_processed_uri),
        artifact_uri=str(text_data["artifact_uri"]),
        start_time=float(text_data["start_time"]),
        end_time=float(text_data["end_time"]),
    )


def load_split_records(split_uri: str, split_name: str) -> list[TrainingRecord]:
    split_path = resolve_uri_to_path(split_uri)
    if not Path(split_path).exists():
        raise FileNotFoundError(f"{split_name} split not found: {split_path}")

    raw_records = load_json(split_path)
    if not isinstance(raw_records, list):
        raise ValueError(f"{split_name} split must be a JSON list.")

    return [load_training_record(raw, split_name, index) for index, raw in enumerate(raw_records)]


def ensure_single_artifact_uri(records: list[TrainingRecord]) -> str:
    artifact_uris = {record.artifact_uri for record in records}
    if len(artifact_uris) != 1:
        raise ValueError(f"training requires a single shared text artifact, found: {sorted(artifact_uris)}")
    return next(iter(artifact_uris))


def iter_batches(records: list[TrainingRecord], batch_size: int, shuffle: bool):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    items = list(records)
    if shuffle:
        random.shuffle(items)

    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def build_targets(
    anchor_times: torch.Tensor,
    gold_start: float,
    gold_end: float,
    positive_iou: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if anchor_times.ndim != 2 or anchor_times.shape[1] != 2:
        raise ValueError("anchor_times must have shape [num_candidates, 2]")

    ious = torch.tensor(
        [
            compute_iou((float(anchor[0].item()), float(anchor[1].item())), (gold_start, gold_end))
            for anchor in anchor_times
        ],
        dtype=torch.float32,
    )
    positive_mask = ious >= positive_iou
    if positive_mask.numel() and not bool(positive_mask.any()):
        positive_mask[int(torch.argmax(ious).item())] = True

    regression_targets = torch.stack(
        [
            torch.full_like(anchor_times[:, 0], float(gold_start)) - anchor_times[:, 0],
            torch.full_like(anchor_times[:, 1], float(gold_end)) - anchor_times[:, 1],
        ],
        dim=1,
    )
    return positive_mask.to(torch.float32), regression_targets, positive_mask


def compute_sample_loss(
    model: CTRLTemporalLocalizer,
    record: TrainingRecord,
    hyperparams: HyperParams,
    device: torch.device,
) -> torch.Tensor:
    visual_features = load_visual_features(record.video_features_uri)
    candidate_features, anchor_times = build_candidate_windows(
        visual_features,
        max_window_scales=hyperparams.max_window_scales,
        context_size=hyperparams.context_size,
        sample_every_sec=hyperparams.sample_every_sec,
    )
    if candidate_features.shape[0] == 0:
        raise TrainingError(f"video has no candidate windows: {record.video_features_uri}")

    text_processed = load_text_processed(record.text_processed_uri)
    sentence_embedding = sentence_embedding_from_text_processed(text_processed).to(device)

    candidate_features = candidate_features.to(device)
    logits, start_offsets, end_offsets = model(candidate_features, sentence_embedding)

    cls_targets, reg_targets, positive_mask = build_targets(
        anchor_times=anchor_times,
        gold_start=record.start_time,
        gold_end=record.end_time,
        positive_iou=hyperparams.positive_iou,
    )
    cls_targets = cls_targets.to(device)
    reg_targets = reg_targets.to(device)
    positive_mask_device = positive_mask.to(device)

    positive_count = int(positive_mask.sum().item())
    negative_count = int(cls_targets.numel()) - positive_count
    pos_weight = logits.new_tensor(max(1.0, negative_count / float(max(positive_count, 1))))

    cls_loss = F.binary_cross_entropy_with_logits(logits, cls_targets, pos_weight=pos_weight)

    if positive_count > 0:
        predicted_offsets = torch.stack([start_offsets, end_offsets], dim=1)
        reg_loss = F.smooth_l1_loss(
            predicted_offsets[positive_mask_device],
            reg_targets[positive_mask_device],
        )
    else:
        reg_loss = logits.new_tensor(0.0)

    return cls_loss + (hyperparams.regression_weight * reg_loss)


def evaluate(
    model: CTRLTemporalLocalizer,
    records: list[TrainingRecord],
    hyperparams: HyperParams,
    device: torch.device,
) -> float:
    if not records:
        return 0.0

    was_training = model.training
    model.eval()
    hits = 0

    with torch.no_grad():
        for record in records:
            visual_features = load_visual_features(record.video_features_uri)
            candidate_features, anchor_times = build_candidate_windows(
                visual_features,
                max_window_scales=model.config.max_window_scales,
                context_size=model.config.context_size,
                sample_every_sec=model.config.sample_every_sec,
            )
            if candidate_features.shape[0] == 0:
                continue

            text_processed = load_text_processed(record.text_processed_uri)
            sentence_embedding = sentence_embedding_from_text_processed(text_processed).to(device)
            logits, start_offsets, end_offsets = model(candidate_features.to(device), sentence_embedding)

            predictions = decode_predictions(
                logits=logits,
                start_offsets=start_offsets,
                end_offsets=end_offsets,
                anchor_times=anchor_times,
                duration_sec=float(visual_features.shape[0] * model.config.sample_every_sec),
                top_n=hyperparams.top_n_eval,
                nms_threshold=model.config.nms_threshold,
            )

            best_iou = 0.0
            for prediction in predictions:
                best_iou = max(
                    best_iou,
                    compute_iou(
                        (prediction["start_sec"], prediction["end_sec"]),
                        (record.start_time, record.end_time),
                    ),
                )
            hits += int(best_iou >= hyperparams.positive_iou)

    if was_training:
        model.train()

    return hits / float(len(records))


def persist_training_artifacts(
    model: CTRLTemporalLocalizer,
    optimizer: torch.optim.Optimizer,
    output_model_uri: str,
    metrics: dict,
) -> tuple[str, str]:
    output_model_path = resolve_uri_to_path(output_model_uri)
    Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model_path)

    optim_path = output_model_path + ".optim"
    torch.save(optimizer.state_dict(), optim_path)

    metrics_path = output_model_path + ".metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    return optim_path, metrics_path


def train_loop(
    train_records: list[TrainingRecord],
    val_records: list[TrainingRecord],
    output_model_uri: str,
    hyperparams: HyperParams,
) -> dict:
    if not train_records:
        raise TrainingError("train split is empty")

    combined_records = train_records + val_records
    artifact_uri = ensure_single_artifact_uri(combined_records or train_records)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    first_features = load_visual_features(train_records[0].video_features_uri)
    first_text = load_text_processed(train_records[0].text_processed_uri)
    first_sentence_embedding = sentence_embedding_from_text_processed(first_text)
    model = CTRLTemporalLocalizer(
        CTRLConfig(
            visual_feature_size=int(first_features.shape[1] * 3),
            sentence_embedding_size=int(first_sentence_embedding.shape[0]),
            semantic_size=hyperparams.semantic_size,
            mlp_hidden_size=hyperparams.mlp_hidden_size,
            max_window_scales=hyperparams.max_window_scales,
            context_size=hyperparams.context_size,
            sample_every_sec=hyperparams.sample_every_sec,
            nms_threshold=hyperparams.nms_threshold,
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.lr)

    metrics = {
        "train_loss": [],
        "val_score": [],
        "best_val_score": None,
        "iterations_completed": 0,
        "device": str(device),
        "artifact_uri": artifact_uri,
        "visual_feature_size": model.config.visual_feature_size,
        "sentence_embedding_size": model.config.sentence_embedding_size,
        "sample_every_sec": model.config.sample_every_sec,
    }

    best_val_score = float("-inf")
    patience_counter = 0
    num_trial = 0
    iteration = 0
    train_started_at = time.time()
    should_stop = False

    while iteration < hyperparams.max_iter and not should_stop:
        for batch_records in iter_batches(train_records, hyperparams.batch_size, shuffle=True):
            optimizer.zero_grad()
            losses = [
                compute_sample_loss(model, record, hyperparams, device)
                for record in batch_records
            ]
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % hyperparams.log_every == 0 or iteration == 1:
                metrics["train_loss"].append({"iteration": iteration, "loss": float(batch_loss.item())})

            if iteration % hyperparams.valid_niter == 0 or iteration == hyperparams.max_iter:
                val_score = evaluate(model, val_records or train_records, hyperparams, device)
                metrics["val_score"].append({"iteration": iteration, "score": val_score})

                if val_score > best_val_score:
                    best_val_score = val_score
                    patience_counter = 0
                    metrics["best_val_score"] = val_score
                    persist_training_artifacts(model, optimizer, output_model_uri, metrics)
                else:
                    patience_counter += 1
                    if patience_counter >= hyperparams.patience:
                        num_trial += 1
                        if num_trial >= hyperparams.max_num_trial:
                            should_stop = True
                            break
                        for group in optimizer.param_groups:
                            group["lr"] *= hyperparams.lr_decay
                        patience_counter = 0

            if iteration >= hyperparams.max_iter:
                break

    metrics["iterations_completed"] = iteration
    metrics["elapsed_sec"] = time.time() - train_started_at

    if metrics["best_val_score"] is None:
        final_score = evaluate(model, val_records or train_records, hyperparams, device)
        metrics["best_val_score"] = final_score
        metrics["val_score"].append({"iteration": iteration, "score": final_score})
    persist_training_artifacts(model, optimizer, output_model_uri, metrics)

    return {
        "status": "completed",
        "output_model_uri": output_model_uri,
        "optimizer_uri": output_model_uri + ".optim",
        "metrics_uri": output_model_uri + ".metrics.json",
        "best_val_score": metrics["best_val_score"],
        "train_size": len(train_records),
        "val_size": len(val_records),
        "artifact_uri": artifact_uri,
    }


def run_training_job(
    train_split_uri: str,
    val_split_uri: str,
    features_root_uri: str,
    output_model_uri: str,
    hyperparams: HyperParams,
) -> dict:
    del features_root_uri
    train_records = load_split_records(train_split_uri, split_name="train")
    val_records = load_split_records(val_split_uri, split_name="val")
    return train_loop(
        train_records=train_records,
        val_records=val_records,
        output_model_uri=output_model_uri,
        hyperparams=hyperparams,
    )


def run_training_from_artifacts(
    video_features_uri: str,
    text_processed_uri: str,
    output_model_uri: str,
    hyperparams: HyperParams,
) -> dict:
    text_data = validate_training_text_processed(text_processed_uri)
    record = TrainingRecord(
        base_name=Path(resolve_uri_to_path(video_features_uri)).stem,
        video_features_uri=video_features_uri,
        text_processed_uri=text_processed_uri,
        artifact_uri=str(text_data["artifact_uri"]),
        start_time=float(text_data["start_time"]),
        end_time=float(text_data["end_time"]),
    )
    return train_loop(
        train_records=[record],
        val_records=[record],
        output_model_uri=output_model_uri,
        hyperparams=hyperparams,
    )
