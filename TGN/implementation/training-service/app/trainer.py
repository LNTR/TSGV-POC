import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_, xavier_normal_

from .model.tgn import TGN


EPS = 1e-6


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


@dataclass
class HyperParams:
    k_scales: int
    delta: int
    threshold: float
    batch_size: int
    lr: float
    hidden_size_textual_lstm: int = 512
    hidden_size_visual_lstm: int = 512
    hidden_size_ilstm: int = 512
    word_embed_size: int = 50
    visual_feature_size: int = 4096
    log_every: int = 10
    max_iter: int = 10000
    valid_niter: int = 50
    top_n_eval: int = 1
    patience: int = 2
    max_num_trial: int = 3
    lr_decay: float = 0.5
    fps: int = 30
    sample_rate: int = 150


@dataclass(frozen=True)
class TextArtifact:
    artifact_uri: str
    word2id: dict[str, int]
    embedding_matrix: np.ndarray


@dataclass(frozen=True)
class TrainingRecord:
    base_name: str
    video_features_uri: str
    text_processed_uri: str
    artifact_uri: str
    token_ids: tuple[int, ...]
    length_t: int
    start_time: float
    end_time: float


@dataclass
class Batch:
    records: list[TrainingRecord]
    visual_features: list[torch.Tensor]
    text_token_ids: list[list[int]]
    lengths_t: list[int]
    labels: torch.Tensor | None


class ManifestDataset:
    def __init__(self, name: str, records: list[TrainingRecord], hyperparams: HyperParams):
        self.name = name
        self.records = records
        self.hyperparams = hyperparams

    def __len__(self) -> int:
        return len(self.records)

    def iter_batches(self, batch_size: int, shuffle: bool, include_labels: bool) -> list[Batch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        indices = np.arange(len(self.records))
        if shuffle and len(indices) > 1:
            np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_records = [self.records[idx] for idx in batch_indices]
            visual_features = [load_visual_features(record.video_features_uri) for record in batch_records]
            text_token_ids = [list(record.token_ids) for record in batch_records]
            lengths_t = [record.length_t for record in batch_records]
            labels = None
            if include_labels:
                labels = pad_labels([
                    generate_label(feature, record, self.hyperparams)
                    for feature, record in zip(visual_features, batch_records)
                ])
            yield Batch(
                records=batch_records,
                visual_features=visual_features,
                text_token_ids=text_token_ids,
                lengths_t=lengths_t,
                labels=labels,
            )


class TrainingError(RuntimeError):
    pass


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=16)
def load_text_artifact(artifact_uri: str) -> TextArtifact:
    root = Path(resolve_uri_to_path(artifact_uri))
    if not root.exists():
        raise FileNotFoundError(f"text artifact path not found: {root}")
    if not root.is_dir():
        raise ValueError("artifact_uri must point to a directory with vocab.json and embeddings.npy")

    vocab_path = root / "vocab.json"
    embeddings_path = root / "embeddings.npy"
    if not vocab_path.exists():
        raise FileNotFoundError(f"missing vocab file: {vocab_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"missing embeddings file: {embeddings_path}")

    raw_vocab = json.loads(vocab_path.read_text())
    if isinstance(raw_vocab, dict):
        word2id = {str(key): int(value) for key, value in raw_vocab.items()}
    elif isinstance(raw_vocab, list):
        word2id = {str(word): idx for idx, word in enumerate(raw_vocab)}
    else:
        raise ValueError("vocab.json must be a dict[word->id] or list[word].")

    embedding_matrix = np.load(embeddings_path)
    if embedding_matrix.ndim != 2:
        raise ValueError("embeddings.npy must be rank-2 [vocab_size, embed_dim].")

    max_id = max(word2id.values()) if word2id else -1
    if max_id >= embedding_matrix.shape[0]:
        raise ValueError("embedding matrix row count is smaller than max vocab id.")

    return TextArtifact(
        artifact_uri=artifact_uri,
        word2id=word2id,
        embedding_matrix=embedding_matrix,
    )


def validate_training_text_processed(text_processed_uri: str) -> dict:
    text_processed_path = resolve_uri_to_path(text_processed_uri)
    if not os.path.exists(text_processed_path):
        raise FileNotFoundError(f"text processed file not found: {text_processed_path}")

    data = load_json(text_processed_path)
    if not isinstance(data, dict):
        raise ValueError(f"text processed file must contain a JSON object: {text_processed_path}")
    if "artifact_uri" not in data:
        raise ValueError(f"text processed file must contain 'artifact_uri': {text_processed_path}")
    if "token_ids" not in data:
        raise ValueError(f"text processed file must contain 'token_ids': {text_processed_path}")

    start_time = data.get("start_time")
    end_time = data.get("end_time")
    if start_time is None or end_time is None:
        raise ValueError(
            "training requires 'start_time' and 'end_time' in text processed files: "
            f"{text_processed_path}"
        )

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

    token_ids = text_data.get("token_ids", [])
    if not isinstance(token_ids, list):
        raise ValueError(f"token_ids must be a list in {text_processed_uri}")
    length_t = int(text_data.get("length_t", len(token_ids)))
    if length_t <= 0:
        length_t = max(len(token_ids), 1)

    return TrainingRecord(
        base_name=str(raw.get("base_name") or text_data.get("base_name") or Path(resolve_uri_to_path(video_features_uri)).stem),
        video_features_uri=str(video_features_uri),
        text_processed_uri=str(text_processed_uri),
        artifact_uri=str(text_data["artifact_uri"]),
        token_ids=tuple(int(token_id) for token_id in token_ids[:length_t]),
        length_t=length_t,
        start_time=float(text_data["start_time"]),
        end_time=float(text_data["end_time"]),
    )


def load_split_records(split_uri: str, split_name: str) -> list[TrainingRecord]:
    split_path = resolve_uri_to_path(split_uri)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"{split_name} split not found: {split_path}")

    raw_records = load_json(split_path)
    if not isinstance(raw_records, list):
        raise ValueError(f"{split_name} split must be a JSON list.")

    return [load_training_record(raw, split_name, index) for index, raw in enumerate(raw_records)]


def ensure_single_artifact(records: list[TrainingRecord]) -> TextArtifact:
    artifact_uris = {record.artifact_uri for record in records}
    if len(artifact_uris) != 1:
        raise ValueError(f"training requires a single shared text artifact, found: {sorted(artifact_uris)}")
    return load_text_artifact(next(iter(artifact_uris)))


def pad_token_ids(token_id_seqs: list[list[int]], pad_id: int) -> torch.Tensor:
    if not token_id_seqs:
        return torch.zeros(0, 0, dtype=torch.long)

    max_len = max(max(len(seq), 1) for seq in token_id_seqs)
    padded = []
    for seq in token_id_seqs:
        safe = list(seq) or [pad_id]
        padded.append(safe + [pad_id] * (max_len - len(safe)))
    return torch.tensor(padded, dtype=torch.long)


def pad_labels(labels: list[torch.Tensor]) -> torch.Tensor:
    if not labels:
        return torch.zeros(0, 0, 0, dtype=torch.float32)

    max_len = max(label.shape[0] for label in labels)
    k_scales = labels[0].shape[1]
    padded = torch.zeros((len(labels), max_len, k_scales), dtype=torch.float32)
    for index, label in enumerate(labels):
        padded[index, :label.shape[0], :] = label.to(torch.float32)
    return padded


def compute_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    if end_a < start_b or end_b < start_a:
        return 0.0
    overlap_start = max(start_a, start_b)
    overlap_end = min(end_a, end_b)
    return max(0.0, overlap_end - overlap_start)


def generate_label(visual_feature: torch.Tensor, record: TrainingRecord, hyperparams: HyperParams) -> torch.Tensor:
    timesteps = visual_feature.shape[0]
    label = torch.zeros((timesteps, hyperparams.k_scales), dtype=torch.float32)

    for t in range(timesteps):
        end_time = t * hyperparams.sample_rate / hyperparams.fps
        for k in range(hyperparams.k_scales):
            start_time = (t - (k + 1) * hyperparams.delta) * hyperparams.sample_rate / hyperparams.fps
            if compute_overlap(start_time, end_time, record.start_time, record.end_time) > hyperparams.threshold:
                label[t, k] = 1.0

    return label


def compute_bce_weights(dataset: ManifestDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    w0 = torch.zeros((dataset.hyperparams.k_scales,), dtype=torch.float32)
    total_time_steps = 0

    for record in dataset.records:
        label = generate_label(load_visual_features(record.video_features_uri), record, dataset.hyperparams)
        total_time_steps += label.shape[0]
        positives = torch.sum(label, dim=0)
        w0 += label.shape[0] - positives

    if total_time_steps == 0:
        raise TrainingError("cannot compute BCE weights on an empty training dataset")

    w0 = (w0 / float(total_time_steps)).to(device)
    return w0, (1.0 - w0).to(device)


def top_n_iou(
    y_pred: torch.Tensor,
    gold_start_times: list[float],
    gold_end_times: list[float],
    hyperparams: HyperParams,
) -> int:
    if y_pred.ndim != 3:
        raise ValueError("expected predictions with shape [batch, T, K]")

    n_batch, _, k_scales = y_pred.shape
    flat = y_pred.view(n_batch, -1)
    k = min(hyperparams.top_n_eval, flat.shape[1])
    if k <= 0:
        return 0

    _, indices = torch.topk(flat, k=k, dim=-1)
    end_times = (indices // k_scales) * hyperparams.sample_rate / hyperparams.fps
    scale_nums = (indices % k_scales) + 1
    start_times = end_times - (scale_nums * hyperparams.delta * hyperparams.sample_rate / hyperparams.fps)

    score = 0
    for batch_idx in range(n_batch):
        max_overlap = 0.0
        for start_time, end_time in zip(start_times[batch_idx], end_times[batch_idx]):
            overlap = compute_overlap(
                float(start_time.item()),
                float(end_time.item()),
                gold_start_times[batch_idx],
                gold_end_times[batch_idx],
            )
            max_overlap = max(max_overlap, overlap)
        score += int(max_overlap > hyperparams.threshold)

    return score


def prepare_embedding(text_artifact: TextArtifact, device: torch.device) -> tuple[nn.Embedding, int]:
    pad_id = int(text_artifact.word2id.get("<pad>", 0))
    embedding = nn.Embedding(
        num_embeddings=text_artifact.embedding_matrix.shape[0],
        embedding_dim=text_artifact.embedding_matrix.shape[1],
        padding_idx=pad_id,
    )
    embedding.weight = nn.Parameter(
        torch.from_numpy(text_artifact.embedding_matrix).to(torch.float32),
        requires_grad=False,
    )
    embedding = embedding.to(device)
    return embedding, pad_id


def build_model(hyperparams: HyperParams, text_artifact: TextArtifact, visual_feature_size: int, device: torch.device) -> TGN:
    model = TGN(
        word_embed_size=int(text_artifact.embedding_matrix.shape[1]),
        hidden_size_textual=hyperparams.hidden_size_textual_lstm,
        hidden_size_visual=hyperparams.hidden_size_visual_lstm,
        hidden_size_ilstm=hyperparams.hidden_size_ilstm,
        k_scales=hyperparams.k_scales,
        visual_feature_size=visual_feature_size,
    )

    for parameter in model.parameters():
        if parameter.requires_grad:
            if parameter.data.ndim > 1:
                xavier_normal_(parameter.data)
            else:
                normal_(parameter.data)

    return model.to(device)


def evaluate(
    model: TGN,
    dataset: ManifestDataset,
    embedding: nn.Embedding,
    pad_id: int,
    hyperparams: HyperParams,
    device: torch.device,
) -> float:
    if len(dataset) == 0:
        return 0.0

    was_training = model.training
    model.eval()
    cumulative_score = 0
    cumulative_samples = 0

    with torch.no_grad():
        for batch in dataset.iter_batches(hyperparams.batch_size, shuffle=False, include_labels=False):
            token_ids = pad_token_ids(batch.text_token_ids, pad_id=pad_id).to(device)
            textual_input = embedding(token_ids)
            lengths_t = [min(length, token_ids.shape[1]) for length in batch.lengths_t]
            probs, mask = model(features_v=batch.visual_features, textual_input=textual_input, lengths_t=lengths_t)
            score = top_n_iou(
                (probs * mask).detach().cpu(),
                [record.start_time for record in batch.records],
                [record.end_time for record in batch.records],
                hyperparams,
            )
            cumulative_score += score
            cumulative_samples += len(batch.records)

    if was_training:
        model.train()

    if cumulative_samples == 0:
        return 0.0
    return cumulative_score / float(cumulative_samples)


def persist_training_artifacts(
    model: TGN,
    optimizer: torch.optim.Optimizer,
    output_model_uri: str,
    metrics: dict,
) -> tuple[str, str]:
    output_model_path = resolve_uri_to_path(output_model_uri)
    os.makedirs(os.path.dirname(output_model_path) or ".", exist_ok=True)
    model.save(output_model_path)

    optim_path = output_model_path + ".optim"
    torch.save(optimizer.state_dict(), optim_path)

    metrics_path = output_model_path + ".metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return optim_path, metrics_path


def train_loop(
    train_dataset: ManifestDataset,
    val_dataset: ManifestDataset,
    output_model_uri: str,
    hyperparams: HyperParams,
) -> dict:
    if len(train_dataset) == 0:
        raise TrainingError("train split is empty")

    combined_records = train_dataset.records + val_dataset.records
    text_artifact = ensure_single_artifact(combined_records)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    visual_feature_size = int(load_visual_features(train_dataset.records[0].video_features_uri).shape[1])
    embedding, pad_id = prepare_embedding(text_artifact, device)
    model = build_model(hyperparams, text_artifact, visual_feature_size, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.lr, betas=(0.5, 0.999))
    w0, w1 = compute_bce_weights(train_dataset, device)

    metrics = {
        "train_loss": [],
        "val_score": [],
        "best_val_score": None,
        "iterations_completed": 0,
        "device": str(device),
        "artifact_uri": text_artifact.artifact_uri,
        "visual_feature_size": visual_feature_size,
        "word_embed_size": int(text_artifact.embedding_matrix.shape[1]),
    }

    best_val_score = None
    patience = 0
    num_trial = 0
    iteration = 0
    report_loss = 0.0
    report_samples = 0
    cumulative_loss = 0.0
    cumulative_samples = 0
    train_time = time.time()

    model.train()

    while iteration < hyperparams.max_iter:
        for batch in train_dataset.iter_batches(hyperparams.batch_size, shuffle=True, include_labels=True):
            token_ids = pad_token_ids(batch.text_token_ids, pad_id=pad_id).to(device)
            textual_input = embedding(token_ids)
            lengths_t = [min(length, token_ids.shape[1]) for length in batch.lengths_t]
            labels = batch.labels.to(device)

            optimizer.zero_grad()
            probs, mask = model(features_v=batch.visual_features, textual_input=textual_input, lengths_t=lengths_t)
            probs = probs.clamp(min=EPS, max=1.0 - EPS)

            batch_loss = -torch.sum((w0 * labels * torch.log(probs) + w1 * (1.0 - labels) * torch.log(1.0 - probs)) * mask)
            batch_loss.backward()
            optimizer.step()

            batch_loss_value = float(batch_loss.item())
            sample_count = len(batch.records)
            iteration += 1
            report_loss += batch_loss_value
            report_samples += sample_count
            cumulative_loss += batch_loss_value
            cumulative_samples += sample_count
            metrics["iterations_completed"] = iteration

            if iteration == 1 or iteration % hyperparams.log_every == 0:
                samples_per_sec = 0.0
                elapsed = max(time.time() - train_time, EPS)
                if report_samples > 0:
                    samples_per_sec = report_samples / elapsed
                avg_loss = report_loss / max(report_samples, 1)
                metrics["train_loss"].append({
                    "iteration": iteration,
                    "loss": avg_loss,
                    "samples_per_sec": samples_per_sec,
                })
                report_loss = 0.0
                report_samples = 0
                train_time = time.time()

            should_validate = len(val_dataset) > 0 and (iteration % hyperparams.valid_niter == 0)
            if should_validate:
                val_score = evaluate(model, val_dataset, embedding, pad_id, hyperparams, device)
                metrics["val_score"].append({"iteration": iteration, "score": val_score})
                metrics["last_cumulative_loss"] = cumulative_loss / max(cumulative_samples, 1)
                metrics["last_cumulative_samples"] = cumulative_samples
                cumulative_loss = 0.0
                cumulative_samples = 0

                is_better = best_val_score is None or val_score > best_val_score
                if is_better:
                    best_val_score = val_score
                    metrics["best_val_score"] = val_score
                    persist_training_artifacts(model, optimizer, output_model_uri, metrics)
                    patience = 0
                elif patience < hyperparams.patience:
                    patience += 1
                    if patience >= hyperparams.patience:
                        num_trial += 1
                        if num_trial >= hyperparams.max_num_trial:
                            iteration = hyperparams.max_iter
                            break

                        checkpoint_path = resolve_uri_to_path(output_model_uri)
                        optimizer_path = checkpoint_path + ".optim"
                        if os.path.exists(checkpoint_path):
                            params = torch.load(checkpoint_path, map_location="cpu")
                            model.load_state_dict(params["state_dict"])
                            model = model.to(device)
                        if os.path.exists(optimizer_path):
                            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

                        for param_group in optimizer.param_groups:
                            param_group["lr"] *= hyperparams.lr_decay

                        patience = 0

            if iteration >= hyperparams.max_iter:
                break

    if best_val_score is None:
        if len(val_dataset) > 0:
            best_val_score = evaluate(model, val_dataset, embedding, pad_id, hyperparams, device)
            metrics["val_score"].append({"iteration": iteration, "score": best_val_score})
            metrics["best_val_score"] = best_val_score
        _, _ = persist_training_artifacts(model, optimizer, output_model_uri, metrics)
    else:
        _, _ = persist_training_artifacts(model, optimizer, output_model_uri, metrics)

    return {
        "model_uri": output_model_uri,
        "optimizer_uri": output_model_uri + ".optim",
        "metrics_uri": output_model_uri + ".metrics.json",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "artifact_uri": text_artifact.artifact_uri,
        "device": str(device),
        "iterations_completed": iteration,
        "best_val_score": best_val_score,
        "visual_feature_size": visual_feature_size,
        "word_embed_size": int(text_artifact.embedding_matrix.shape[1]),
    }


def run_training_job(
    train_split_uri: str,
    val_split_uri: str,
    features_root_uri: str,
    output_model_uri: str,
    hyperparams: HyperParams,
) -> dict:
    train_records = load_split_records(train_split_uri, "train")
    val_records = load_split_records(val_split_uri, "val")

    metadata = train_loop(
        train_dataset=ManifestDataset(name="train", records=train_records, hyperparams=hyperparams),
        val_dataset=ManifestDataset(name="val", records=val_records, hyperparams=hyperparams),
        output_model_uri=output_model_uri,
        hyperparams=hyperparams,
    )
    metadata["features_root_uri"] = features_root_uri
    return metadata


def run_training_from_artifacts(
    video_features_uri: str,
    text_processed_uri: str,
    output_model_uri: str,
    hyperparams: HyperParams,
) -> dict:
    record = load_training_record(
        {
            "base_name": Path(resolve_uri_to_path(video_features_uri)).stem,
            "video_features_uri": video_features_uri,
            "text_processed_uri": text_processed_uri,
        },
        split_name="artifact",
        index=0,
    )

    return train_loop(
        train_dataset=ManifestDataset(name="artifact-train", records=[record], hyperparams=hyperparams),
        val_dataset=ManifestDataset(name="artifact-val", records=[], hyperparams=hyperparams),
        output_model_uri=output_model_uri,
        hyperparams=hyperparams,
    )
