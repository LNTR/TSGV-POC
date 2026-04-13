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
    batch_size: int = Field(..., ge=1)
    lr: float
    semantic_size: int = Field(default=1024, ge=1)
    mlp_hidden_size: int = Field(default=1000, ge=1)
    max_window_scales: int = Field(default=8, ge=1)
    context_size: int = Field(default=1, ge=1)
    positive_iou: float = Field(default=0.5, gt=0.0, le=1.0)
    regression_weight: float = Field(default=0.01, ge=0.0)
    sample_every_sec: float = Field(default=5.0, gt=0.0)
    nms_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    log_every: int = Field(default=10, ge=1)
    max_iter: int = Field(default=1000, ge=1)
    valid_niter: int = Field(default=50, ge=1)
    top_n_eval: int = Field(default=5, ge=1)
    patience: int = Field(default=2, ge=0)
    max_num_trial: int = Field(default=3, ge=1)
    lr_decay: float = Field(default=0.5, gt=0.0)


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


def parse_hyperparams(raw_hyperparams: dict | None) -> HyperParams:
    raw = raw_hyperparams or {}
    sample_every_sec = raw.get("sample_every_sec")
    if sample_every_sec is None and raw.get("sample_rate") and raw.get("fps"):
        sample_every_sec = float(raw["sample_rate"]) / float(raw["fps"])

    return HyperParams(
        batch_size=int(raw.get("batch_size", 1)),
        lr=float(raw.get("lr", 0.001)),
        semantic_size=int(raw.get("semantic_size", 1024)),
        mlp_hidden_size=int(raw.get("mlp_hidden_size", 1000)),
        max_window_scales=int(raw.get("max_window_scales", raw.get("K", 8))),
        context_size=int(raw.get("context_size", 1)),
        positive_iou=float(raw.get("positive_iou", raw.get("threshold", 0.5))),
        regression_weight=float(raw.get("regression_weight", 0.01)),
        sample_every_sec=float(sample_every_sec or 5.0),
        nms_threshold=float(raw.get("nms_threshold", 0.45)),
        log_every=int(raw.get("log_every", 10)),
        max_iter=int(raw.get("max_iter", 1000)),
        valid_niter=int(raw.get("valid_niter", 50)),
        top_n_eval=int(raw.get("top_n_eval", 5)),
        patience=int(raw.get("patience", 2)),
        max_num_trial=int(raw.get("max_num_trial", 3)),
        lr_decay=float(raw.get("lr_decay", 0.5)),
    )


app = FastAPI(
    title="TALL Training Service",
    version="0.1.0",
    lifespan=build_service_lifespan(
        logical_service="training-service",
        default_port=8004,
        service_version="0.1.0",
        api_title="TALL Training Service",
    ),
)


@app.get("/registry/self")
def registry_self() -> dict:
    return app.state.service_registry


@app.post("/jobs/train", response_model=JobAccepted, status_code=202)
def create_training_job(payload: TrainRequest) -> JobAccepted:
    if payload.model_family not in {None, "tall"}:
        raise HTTPException(status_code=400, detail="TALL training service only supports model_family='tall'")
    hp = HyperParams(
        batch_size=payload.hyperparams.batch_size,
        lr=payload.hyperparams.lr,
        semantic_size=payload.hyperparams.semantic_size,
        mlp_hidden_size=payload.hyperparams.mlp_hidden_size,
        max_window_scales=payload.hyperparams.max_window_scales,
        context_size=payload.hyperparams.context_size,
        positive_iou=payload.hyperparams.positive_iou,
        regression_weight=payload.hyperparams.regression_weight,
        sample_every_sec=payload.hyperparams.sample_every_sec,
        nms_threshold=payload.hyperparams.nms_threshold,
        log_every=payload.hyperparams.log_every,
        max_iter=payload.hyperparams.max_iter,
        valid_niter=payload.hyperparams.valid_niter,
        top_n_eval=payload.hyperparams.top_n_eval,
        patience=payload.hyperparams.patience,
        max_num_trial=payload.hyperparams.max_num_trial,
        lr_decay=payload.hyperparams.lr_decay,
    )

    metadata = run_training_job(
        train_split_uri=payload.train_split_uri,
        val_split_uri=payload.val_split_uri,
        features_root_uri=payload.features_root_uri,
        output_model_uri=payload.output_model_uri,
        hyperparams=hp,
    )
    return JobAccepted(status="accepted", metadata=metadata)


@app.post("/jobs/train-from-artifacts", response_model=JobAccepted, status_code=202)
def create_training_from_artifacts_job(payload: TrainFromArtifactsRequest) -> JobAccepted:
    if payload.model_family not in {None, "tall"}:
        raise HTTPException(status_code=400, detail="TALL training service only supports model_family='tall'")
    metadata = run_training_from_artifacts(
        video_features_uri=payload.video_features_uri,
        text_processed_uri=payload.text_processed_uri,
        output_model_uri=payload.output_model_uri,
        hyperparams=parse_hyperparams(payload.hyperparams),
    )
    return JobAccepted(status="accepted", metadata=metadata)
