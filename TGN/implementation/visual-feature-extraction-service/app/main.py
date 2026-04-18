import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = next((parent for parent in THIS_FILE.parents if (parent / "service_registry.py").exists()), THIS_FILE.parents[2])
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from service_registry import build_service_lifespan

from .extraction import extract_visual_features, maybe_trigger_downstream


class ServiceModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class NormalizeConfig(ServiceModel):
    mean: list[float] = [0.485, 0.456, 0.406]
    std: list[float] = [0.229, 0.224, 0.225]


class JobEnvelope(ServiceModel):
    job_id: str
    trace_id: str
    dataset: str
    created_at: str
    version: str = "v1"


class FeatureRecipe(ServiceModel):
    strategy: Literal["frame_encoder", "clip_encoder"] | None = None
    encoder: str | None = None


class DownstreamControl(ServiceModel):
    mode: Literal["inference", "training"] | None = None
    service_url: str | None = None
    model_uri: str | None = None
    output_model_uri: str | None = None
    top_n: int | None = Field(default=None, ge=1)
    hyperparams: dict | None = None


class FeatureRequest(ServiceModel):
    job: JobEnvelope | None = None
    frames_uri: str
    feature_recipe: FeatureRecipe | None = None
    downstream: DownstreamControl | None = None
    encoder: str | None = "vgg16"
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
        return self.encoder or "vgg16"

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
    downstream: dict


app = FastAPI(
    title="Visual Feature Extraction Service",
    version="0.1.0",
    lifespan=build_service_lifespan(
        logical_service="visual-feature-extraction-service",
        default_port=8002,
        service_version="0.1.0",
        api_title="Visual Feature Extraction Service",
    ),
)


@app.get("/registry/self")
def registry_self() -> dict:
    return app.state.service_registry


@app.post("/jobs/features", response_model=JobAccepted, status_code=202)
def create_feature_job(payload: FeatureRequest) -> JobAccepted:
    if payload.feature_recipe and payload.feature_recipe.strategy == "clip_encoder":
        raise HTTPException(status_code=400, detail="TGN feature extraction expects frame_encoder feature recipes")

    metadata = extract_visual_features(
        frames_uri=payload.frames_uri,
        mean=payload.normalize.mean,
        std=payload.normalize.std,
        encoder=payload.resolved_encoder(),
    )
    downstream = maybe_trigger_downstream(
        frames_uri=payload.frames_uri,
        features_uri=metadata["features_uri"],
        downstream_mode=payload.resolved_downstream_mode(),
        downstream_service_url=payload.resolved_downstream_service_url(),
        model_uri=payload.resolved_model_uri(),
        output_model_uri=payload.resolved_output_model_uri(),
        artifact_uri=payload.artifact_uri,
        top_n=payload.resolved_top_n(),
        hyperparams=payload.resolved_hyperparams(),
    )
    return JobAccepted(
        status="accepted",
        metadata=metadata,
        downstream=downstream,
    )
