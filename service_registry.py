from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

try:
    from pymongo import MongoClient
except ImportError:  # pragma: no cover - handled gracefully when dependency is absent
    MongoClient = None


LOGGER = logging.getLogger("service_registry")
KNOWN_MODELS = ("tgn", "tall")
KNOWN_LOGICAL_SERVICES = (
    "visual-preprocessing-service",
    "visual-feature-extraction-service",
    "text-processing-service",
    "training-service",
    "inference-service",
    "evaluation-service",
)
COMPONENT_COMPATIBILITY_RULES = {
    "visual-preprocessing-service": {
        "produces_for": ("visual-feature-extraction-service",),
    },
    "visual-feature-extraction-service": {
        "consumed_by": ("training-service", "inference-service", "evaluation-service"),
        "depends_on": ("visual-preprocessing-service",),
    },
    "text-processing-service": {
        "consumed_by": ("training-service", "inference-service", "evaluation-service"),
    },
    "training-service": {
        "depends_on": ("visual-feature-extraction-service", "text-processing-service"),
        "produces_for": ("inference-service", "evaluation-service"),
    },
    "inference-service": {
        "depends_on": ("training-service", "visual-feature-extraction-service", "text-processing-service"),
        "produces_for": ("evaluation-service",),
    },
    "evaluation-service": {
        "depends_on": ("training-service", "inference-service", "visual-feature-extraction-service", "text-processing-service"),
    },
}


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_registry_uri() -> str | None:
    return os.getenv("MONGODB_URI") or os.getenv("SERVICE_REGISTRY_MONGODB_URI")


def _build_default_url(default_port: int) -> str:
    host = os.getenv("SERVICE_SELF_HOST", "127.0.0.1")
    port = int(os.getenv("SERVICE_PORT", str(default_port)))
    scheme = os.getenv("SERVICE_SELF_SCHEME", "http")
    return f"{scheme}://{host}:{port}"


def _build_seeded_catalog() -> dict[str, dict[str, dict[str, Any]]]:
    catalog: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name in KNOWN_MODELS:
        catalog[model_name] = {}
        for logical_service in KNOWN_LOGICAL_SERVICES:
            implementation_name = f"{model_name}-{logical_service}"
            catalog[model_name][logical_service] = {
                "model_name": model_name,
                "logical_service": logical_service,
                "implementation_name": implementation_name,
                "service_id": f"{model_name}:{logical_service}:{implementation_name}",
            }
    return catalog


SEEDED_SERVICE_CATALOG = _build_seeded_catalog()


def _lookup_seeded_service(model_name: str, logical_service: str) -> dict[str, Any] | None:
    seeded_model = SEEDED_SERVICE_CATALOG.get(model_name)
    if seeded_model is None:
        return None
    seeded_entry = seeded_model.get(logical_service)
    if seeded_entry is None:
        return None
    return dict(seeded_entry)


def _build_component_compatibility(
    *,
    logical_service: str,
    compatible_models: list[str],
) -> dict[str, list[dict[str, Any]]]:
    rules = COMPONENT_COMPATIBILITY_RULES.get(logical_service, {})
    component_map: dict[str, list[dict[str, Any]]] = {}

    for relation_name, related_services in rules.items():
        entries: list[dict[str, Any]] = []
        for related_logical_service in related_services:
            for compatible_model in compatible_models:
                seeded_entry = _lookup_seeded_service(compatible_model, related_logical_service)
                if seeded_entry is None:
                    continue
                entries.append(seeded_entry)
        component_map[relation_name] = entries

    return component_map


def _build_compatibility_payload(
    *,
    model_name: str,
    logical_service: str,
    compatible_models: list[str],
    shared_across_models: bool,
) -> dict[str, Any]:
    compatible_services: list[dict[str, Any]] = []
    missing_models: list[str] = []

    for compatible_model in compatible_models:
        seeded_entry = _lookup_seeded_service(compatible_model, logical_service)
        if seeded_entry is None:
            missing_models.append(compatible_model)
            continue
        compatible_services.append(seeded_entry)

    component_compatibility = _build_component_compatibility(
        logical_service=logical_service,
        compatible_models=compatible_models,
    )

    return {
        "seeded_catalog_models": list(KNOWN_MODELS),
        "declared_compatible_models": compatible_models,
        "shared_across_models": shared_across_models,
        "compatible_services": compatible_services,
        "component_compatibility": component_compatibility,
        "missing_seeded_models": missing_models,
        "supports_requested_model": model_name in compatible_models,
        "lookup_mode": "seeded_component_catalog",
    }


def build_registry_document(
    *,
    logical_service: str,
    default_port: int,
    service_version: str,
    api_title: str,
) -> dict[str, Any]:
    model_name = os.getenv("SERVICE_MODEL_NAME", "unknown").strip() or "unknown"
    implementation_name = (
        os.getenv("SERVICE_IMPLEMENTATION_NAME", f"{model_name}-{logical_service}").strip()
        or f"{model_name}-{logical_service}"
    )
    compatible_models = _parse_csv(os.getenv("SERVICE_COMPATIBLE_MODELS")) or [model_name]
    compatible_models = list(dict.fromkeys(compatible_models))
    shared_across_models = _parse_bool(
        os.getenv("SERVICE_SHARED_ACROSS_MODELS"),
        default=len(compatible_models) > 1,
    )
    compatibility = _build_compatibility_payload(
        model_name=model_name,
        logical_service=logical_service,
        compatible_models=compatible_models,
        shared_across_models=shared_across_models,
    )
    endpoint_url = os.getenv("SERVICE_SELF_URL") or _build_default_url(default_port)
    registered_at = datetime.now(timezone.utc).isoformat()

    service_id = os.getenv(
        "SERVICE_REGISTRY_SERVICE_ID",
        f"{model_name}:{logical_service}:{implementation_name}",
    )

    return {
        "service_id": service_id,
        "logical_service": logical_service,
        "model_name": model_name,
        "implementation_name": implementation_name,
        "compatible_models": compatible_models,
        "shared_across_models": shared_across_models,
        "compatibility": compatibility,
        "endpoint": {
            "url": endpoint_url,
            "port": int(os.getenv("SERVICE_PORT", str(default_port))),
            "scheme": os.getenv("SERVICE_SELF_SCHEME", "http"),
        },
        "runtime": {
            "platform": os.getenv("SERVICE_RUNTIME_PLATFORM", "local"),
            "namespace": os.getenv("SERVICE_RUNTIME_NAMESPACE", "default"),
        },
        "metadata": {
            "api_title": api_title,
            "service_version": service_version,
        },
        "status": "ready",
        "registered_at": registered_at,
        "updated_at": registered_at,
    }


def register_service(
    *,
    logical_service: str,
    default_port: int,
    service_version: str,
    api_title: str,
) -> dict[str, Any]:
    document = build_registry_document(
        logical_service=logical_service,
        default_port=default_port,
        service_version=service_version,
        api_title=api_title,
    )
    uri = _get_registry_uri()

    if not _parse_bool(os.getenv("SERVICE_REGISTRY_ENABLED"), default=bool(uri)):
        return {
            "enabled": False,
            "reason": "service_registry_disabled",
            "document": document,
        }

    if not uri:
        return {
            "enabled": False,
            "reason": "missing_mongodb_uri",
            "document": document,
        }

    if MongoClient is None:
        return {
            "enabled": False,
            "reason": "pymongo_not_installed",
            "document": document,
        }

    database_name = os.getenv("SERVICE_REGISTRY_DATABASE", "service_registry")
    collection_name = os.getenv("SERVICE_REGISTRY_COLLECTION", "services")
    timeout_ms = int(os.getenv("SERVICE_REGISTRY_TIMEOUT_MS", "3000"))
    max_attempts = max(1, int(os.getenv("SERVICE_REGISTRY_MAX_ATTEMPTS", "10")))
    backoff_sec = max(0.0, float(os.getenv("SERVICE_REGISTRY_RETRY_BACKOFF_SEC", "1.0")))
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        try:
            collection = client[database_name][collection_name]
            collection.create_index("service_id", unique=True)
            collection.create_index([("logical_service", 1), ("model_name", 1)])
            collection.update_one(
                {"service_id": document["service_id"]},
                {
                    "$set": document,
                    "$setOnInsert": {"created_at": document["registered_at"]},
                },
                upsert=True,
            )
            return {
                "enabled": True,
                "database": database_name,
                "collection": collection_name,
                "document": document,
                "attempts": attempt,
            }
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            last_error = str(exc)
            if attempt < max_attempts:
                LOGGER.warning(
                    "Service registration attempt %s/%s failed for %s: %s",
                    attempt,
                    max_attempts,
                    logical_service,
                    exc,
                )
                time.sleep(backoff_sec)
            else:
                LOGGER.warning("Service registration failed for %s after %s attempts: %s", logical_service, attempt, exc)
        finally:
            client.close()

    return {
        "enabled": False,
        "reason": "mongodb_registration_failed",
        "error": last_error,
        "document": document,
        "attempts": max_attempts,
    }


def build_service_lifespan(
    *,
    logical_service: str,
    default_port: int,
    service_version: str,
    api_title: str,
):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service_registry = register_service(
            logical_service=logical_service,
            default_port=default_port,
            service_version=service_version,
            api_title=api_title,
        )
        yield

    return lifespan
