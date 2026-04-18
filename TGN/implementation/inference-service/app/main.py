import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = next((parent for parent in THIS_FILE.parents if (parent / "service_registry.py").exists()), THIS_FILE.parents[2])
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from service_registry import build_service_lifespan

from .pipeline import infer_segments


class ServiceModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class InferenceRequest(ServiceModel):
    model_uri: str
    video_features_uri: str
    text_processed_uri: str
    top_n: int = Field(default=5, ge=1)


class TemporalSegment(ServiceModel):
    start_sec: float
    end_sec: float
    score: float


class InferenceResponse(ServiceModel):
    segments: list[TemporalSegment]


app = FastAPI(
    title="Inference Service",
    version="0.1.0",
    lifespan=build_service_lifespan(
        logical_service="inference-service",
        default_port=8005,
        service_version="0.1.0",
        api_title="Inference Service",
    ),
)


@app.get("/registry/self")
def registry_self() -> dict:
    return app.state.service_registry


@app.post("/infer/ground", response_model=InferenceResponse)
def infer_grounding(payload: InferenceRequest) -> InferenceResponse:
    segments = infer_segments(
        model_uri=payload.model_uri,
        video_features_uri=payload.video_features_uri,
        text_processed_uri=payload.text_processed_uri,
        top_n=payload.top_n,
    )
    return InferenceResponse(segments=segments)
