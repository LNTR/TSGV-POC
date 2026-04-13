import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[2]
if str(IMPLEMENTATION_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPLEMENTATION_ROOT))

from service_registry import build_service_lifespan

from .evaluator import run_evaluation_job


class ServiceModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class JobEnvelope(ServiceModel):
    job_id: str
    trace_id: str
    dataset: str
    created_at: str
    version: str = "v1"


class EvaluationRequest(ServiceModel):
    job: JobEnvelope
    model_uri: str
    test_split_uri: str
    features_root_uri: str
    metrics: list[str]


class EvaluationResponse(ServiceModel):
    job_id: str
    scores: dict[str, float]


app = FastAPI(
    title="Evaluation Service",
    version="0.1.0",
    lifespan=build_service_lifespan(
        logical_service="evaluation-service",
        default_port=8006,
        service_version="0.1.0",
        api_title="Evaluation Service",
    ),
)


@app.get("/registry/self")
def registry_self() -> dict:
    return app.state.service_registry


@app.post("/jobs/evaluate", response_model=EvaluationResponse)
def evaluate(payload: EvaluationRequest) -> EvaluationResponse:
    scores = run_evaluation_job(
        model_uri=payload.model_uri,
        test_split_uri=payload.test_split_uri,
        metrics=payload.metrics,
        features_root_uri=payload.features_root_uri,
    )

    return EvaluationResponse(job_id=payload.job.job_id, scores=scores)
