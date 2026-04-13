from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

try:
    from pymongo import MongoClient
except ImportError:  # pragma: no cover - handled gracefully when dependency is absent
    MongoClient = None


LOGGER = logging.getLogger("service_registry")


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


def build_registry_document(
    *,
    logical_service: str,
    default_port: int,
    service_version: str,
    api_title: str,
) -> dict[str, Any]:
    model_name = os.getenv("SERVICE_MODEL_NAME", "tgn").strip() or "tgn"
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
        }
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        LOGGER.warning("Service registration failed for %s: %s", logical_service, exc)
        return {
            "enabled": False,
            "reason": "mongodb_registration_failed",
            "error": str(exc),
            "document": document,
        }
    finally:
        client.close()


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
