import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[2]
if str(IMPLEMENTATION_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPLEMENTATION_ROOT))

from service_registry import build_service_lifespan

from .trainer import HyperParams, run_training_from_artifacts, run_training_job


class ServiceModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class JobEnvelope(ServiceModel):
    job_id: str
    trace_id: str
    dataset: str
    created_at: str
    version: str = "v1"


class HyperParametersIn(ServiceModel):
    K: int = Field(..., ge=1)
    delta: int = Field(..., ge=1)
    threshold: float
    batch_size: int = Field(..., ge=1)
    lr: float
    hidden_size_textual_lstm: int = 512
    hidden_size_visual_lstm: int = 512
    hidden_size_ilstm: int = 512
    word_embed_size: int = 50
    visual_feature_size: int = 4096
    log_every: int = Field(default=10, ge=1)
    max_iter: int = Field(default=10000, ge=1)
    valid_niter: int = Field(default=50, ge=1)
    top_n_eval: int = Field(default=1, ge=1)
    patience: int = Field(default=2, ge=0)
    max_num_trial: int = Field(default=3, ge=1)
    lr_decay: float = Field(default=0.5, gt=0.0)
    fps: int = Field(default=30, ge=1)
    sample_rate: int = Field(default=150, ge=1)


class TrainRequest(ServiceModel):
    job: JobEnvelope | None = None
    train_split_uri: str
    val_split_uri: str
    features_root_uri: str
    hyperparams: HyperParametersIn
    output_model_uri: str
    model_family: str | None = None


class JobAccepted(ServiceModel):
    status: str
    metadata: dict


class TrainFromArtifactsRequest(ServiceModel):
    job: JobEnvelope | None = None
    video_features_uri: str
    text_processed_uri: str
    output_model_uri: str
    hyperparams: dict | None = None
    model_family: str | None = None


app = FastAPI(
    title="Training Service",
    version="0.1.0",
    lifespan=build_service_lifespan(
        logical_service="training-service",
        default_port=8004,
        service_version="0.1.0",
        api_title="Training Service",
    ),
)


@app.get("/registry/self")
def registry_self() -> dict:
    return app.state.service_registry


@app.post("/jobs/train", response_model=JobAccepted, status_code=202)
def create_training_job(payload: TrainRequest) -> JobAccepted:
    if payload.model_family not in {None, "tgn"}:
        raise HTTPException(status_code=400, detail="TGN training service only supports model_family='tgn'")
    hp = HyperParams(
        k_scales=payload.hyperparams.K,
        delta=payload.hyperparams.delta,
        threshold=payload.hyperparams.threshold,
        batch_size=payload.hyperparams.batch_size,
        lr=payload.hyperparams.lr,
        hidden_size_textual_lstm=payload.hyperparams.hidden_size_textual_lstm,
        hidden_size_visual_lstm=payload.hyperparams.hidden_size_visual_lstm,
        hidden_size_ilstm=payload.hyperparams.hidden_size_ilstm,
        word_embed_size=payload.hyperparams.word_embed_size,
        visual_feature_size=payload.hyperparams.visual_feature_size,
        log_every=payload.hyperparams.log_every,
        max_iter=payload.hyperparams.max_iter,
        valid_niter=payload.hyperparams.valid_niter,
        top_n_eval=payload.hyperparams.top_n_eval,
        patience=payload.hyperparams.patience,
        max_num_trial=payload.hyperparams.max_num_trial,
        lr_decay=payload.hyperparams.lr_decay,
        fps=payload.hyperparams.fps,
        sample_rate=payload.hyperparams.sample_rate,
    )

    metadata = run_training_job(
        train_split_uri=payload.train_split_uri,
        val_split_uri=payload.val_split_uri,
        features_root_uri=payload.features_root_uri,
        output_model_uri=payload.output_model_uri,
        hyperparams=hp,
    )

    return JobAccepted(
        status="accepted",
        metadata=metadata,
    )


@app.post("/jobs/train-from-artifacts", response_model=JobAccepted, status_code=202)
def create_training_from_artifacts_job(payload: TrainFromArtifactsRequest) -> JobAccepted:
    if payload.model_family not in {None, "tgn"}:
        raise HTTPException(status_code=400, detail="TGN training service only supports model_family='tgn'")
    raw_hyperparams = payload.hyperparams or {}
    hp = HyperParams(
        k_scales=int(raw_hyperparams.get("K", 16)),
        delta=int(raw_hyperparams.get("delta", 2)),
        threshold=float(raw_hyperparams.get("threshold", 0.5)),
        batch_size=int(raw_hyperparams.get("batch_size", 1)),
        lr=float(raw_hyperparams.get("lr", 0.001)),
        hidden_size_textual_lstm=int(raw_hyperparams.get("hidden_size_textual_lstm", 512)),
        hidden_size_visual_lstm=int(raw_hyperparams.get("hidden_size_visual_lstm", 512)),
        hidden_size_ilstm=int(raw_hyperparams.get("hidden_size_ilstm", 512)),
        word_embed_size=int(raw_hyperparams.get("word_embed_size", 50)),
        visual_feature_size=int(raw_hyperparams.get("visual_feature_size", 4096)),
        log_every=int(raw_hyperparams.get("log_every", 10)),
        max_iter=int(raw_hyperparams.get("max_iter", 10000)),
        valid_niter=int(raw_hyperparams.get("valid_niter", 50)),
        top_n_eval=int(raw_hyperparams.get("top_n_eval", 1)),
        patience=int(raw_hyperparams.get("patience", 2)),
        max_num_trial=int(raw_hyperparams.get("max_num_trial", 3)),
        lr_decay=float(raw_hyperparams.get("lr_decay", 0.5)),
        fps=int(raw_hyperparams.get("fps", 30)),
        sample_rate=int(raw_hyperparams.get("sample_rate", 150)),
    )
    metadata = run_training_from_artifacts(
        video_features_uri=payload.video_features_uri,
        text_processed_uri=payload.text_processed_uri,
        output_model_uri=payload.output_model_uri,
        hyperparams=hp,
    )
    return JobAccepted(status="accepted", metadata=metadata)
