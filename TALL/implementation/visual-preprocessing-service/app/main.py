import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[2]
if str(IMPLEMENTATION_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPLEMENTATION_ROOT))

from service_registry import build_service_lifespan

from .processing import (
    build_visual_feature_uri,
    build_visual_preprocess_uri,
    forward_to_feature_service,
    run_preprocess_job,
)


class ServiceModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class NormalizeConfig(ServiceModel):
    mean: list[float] = [0.0, 0.0, 0.0]
    std: list[float] = [1.0, 1.0, 1.0]


class JobEnvelope(ServiceModel):
    job_id: str
    trace_id: str
    dataset: str
    created_at: str
    version: str = "v1"


class FeatureRecipe(ServiceModel):
    strategy: Literal["frame_encoder", "clip_encoder"] | None = None
    encoder: str | None = None
    clip_num_frames: int | None = Field(default=None, ge=1)
    clip_frame_step: int | None = Field(default=None, ge=1)


class DownstreamControl(ServiceModel):
    mode: Literal["inference", "training"] | None = None
    service_url: str | None = None
    model_uri: str | None = None
    output_model_uri: str | None = None
    top_n: int | None = Field(default=None, ge=1)
    hyperparams: dict | None = None


class PreprocessRequest(ServiceModel):
    job: JobEnvelope | None = None
    input_video_uri: str
    output_frame_size: int = Field(..., ge=1)
    sample_every_sec: float = Field(..., gt=0.0)
    feature_service_url: str
    feature_recipe: FeatureRecipe | None = None
    downstream: DownstreamControl | None = None
    encoder: str | None = "c3d"
    clip_num_frames: int = Field(default=16, ge=1)
    clip_frame_step: int = Field(default=1, ge=1)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)
    downstream_mode: Literal["inference", "training"] | None = None
    downstream_service_url: str | None = None
    model_uri: str | None = None
    output_model_uri: str | None = None
    artifact_uri: str | None = None
    top_n: int = Field(default=5, ge=1)
    hyperparams: dict | None = None

    def resolved_encoder(self) -> str:
        if self.feature_recipe and self.feature_recipe.encoder:
            return self.feature_recipe.encoder
        return self.encoder or "c3d"

    def resolved_clip_num_frames(self) -> int:
        if self.feature_recipe and self.feature_recipe.clip_num_frames is not None:
            return self.feature_recipe.clip_num_frames
        return self.clip_num_frames

    def resolved_clip_frame_step(self) -> int:
        if self.feature_recipe and self.feature_recipe.clip_frame_step is not None:
            return self.feature_recipe.clip_frame_step
        return self.clip_frame_step

    def resolved_sample_every_sec(self) -> int:
        return max(1, int(self.sample_every_sec))

    def resolved_downstream_mode(self) -> Literal["inference", "training"] | None:
        if self.downstream and self.downstream.mode is not None:
            return self.downstream.mode
        return self.downstream_mode

    def resolved_downstream_service_url(self) -> str | None:
        if self.downstream and self.downstream.service_url is not None:
            return self.downstream.service_url
        return self.downstream_service_url

    def resolved_model_uri(self) -> str | None:
        if self.downstream and self.downstream.model_uri is not None:
            return self.downstream.model_uri
        return self.model_uri

    def resolved_output_model_uri(self) -> str | None:
        if self.downstream and self.downstream.output_model_uri is not None:
            return self.downstream.output_model_uri
        return self.output_model_uri

    def resolved_top_n(self) -> int:
        if self.downstream and self.downstream.top_n is not None:
            return self.downstream.top_n
        return self.top_n

    def resolved_hyperparams(self) -> dict | None:
        if self.downstream and self.downstream.hyperparams is not None:
            return self.downstream.hyperparams
        return self.hyperparams


class JobAccepted(ServiceModel):
    status: str
    metadata: dict
    forwarded: dict


app = FastAPI(
    title="Visual Preprocessing Service",
    version="0.1.0",
    lifespan=build_service_lifespan(
        logical_service="visual-preprocessing-service",
        default_port=8001,
        service_version="0.1.0",
        api_title="Visual Preprocessing Service",
    ),
)


@app.get("/registry/self")
def registry_self() -> dict:
    return app.state.service_registry


@app.post("/jobs/preprocess", response_model=JobAccepted, status_code=202)
def create_preprocess_job(payload: PreprocessRequest) -> JobAccepted:
    if payload.feature_recipe and payload.feature_recipe.strategy == "frame_encoder":
        raise HTTPException(status_code=400, detail="TALL preprocessing expects clip_encoder feature recipes")

    resolved_sample_every_sec = payload.resolved_sample_every_sec()
    resolved_encoder = payload.resolved_encoder()
    metadata = run_preprocess_job(
        input_video_uri=payload.input_video_uri,
        output_frame_size=payload.output_frame_size,
        sample_every_sec=resolved_sample_every_sec,
        encoder=resolved_encoder,
    )
    forwarded = forward_to_feature_service(
        feature_service_url=payload.feature_service_url,
        input_video_uri=payload.input_video_uri,
        output_frame_size=payload.output_frame_size,
        sample_every_sec=resolved_sample_every_sec,
        encoder=resolved_encoder,
        clip_num_frames=payload.resolved_clip_num_frames(),
        clip_frame_step=payload.resolved_clip_frame_step(),
        normalize=payload.normalize.model_dump(),
        downstream_mode=payload.resolved_downstream_mode(),
        downstream_service_url=payload.resolved_downstream_service_url(),
        model_uri=payload.resolved_model_uri(),
        output_model_uri=payload.resolved_output_model_uri(),
        artifact_uri=payload.artifact_uri,
        top_n=payload.resolved_top_n(),
        hyperparams=payload.resolved_hyperparams(),
    )
    metadata["features_uri"] = build_visual_feature_uri(payload.input_video_uri)
    metadata["frames_uri"] = build_visual_preprocess_uri(payload.input_video_uri)
    return JobAccepted(
        status="accepted",
        metadata=metadata,
        forwarded=forwarded,
    )
