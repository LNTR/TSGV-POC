#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import yaml


SERVICE_FOLDERS = {
    "visual-preprocessing-service": "visual-preprocessing-service",
    "visual-feature-extraction-service": "visual-feature-extraction-service",
    "text-processing-service": "text-processing-service",
    "training-service": "training-service",
    "inference-service": "inference-service",
    "evaluation-service": "evaluation-service",
}
ENDPOINT_PEER_MAP = {
    "/jobs/features": "visual-feature-extraction-service",
    "/infer/ground": "inference-service",
    "/jobs/train-from-artifacts": "training-service",
    "/jobs/train": "training-service",
}
INFRA_ENV_KEYS = {
    "MONGODB_URI": "mongodb",
    "REDIS_URL": "redis",
}
EXPLICIT_SERVICE_PATTERN = re.compile(r"http://([a-z0-9-]+):\d+")
SHARED_URI_PATTERN = re.compile(r"shared://([a-z0-9_.-]+)(?:/([a-z0-9_.-]+))?")


def discover_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_docker_compose_services(compose_path: Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(compose_path.read_text()) or {}
    services = payload.get("services", {})
    if not isinstance(services, dict):
        return {}
    return services


def iter_python_files(service_root: Path) -> list[Path]:
    return sorted(path for path in service_root.rglob("*.py") if "__pycache__" not in path.parts)


def collect_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def normalize_shared_uri(match: re.Match[str]) -> str:
    first = match.group(1)
    second = match.group(2)
    if second:
        return f"shared://{first}/{second}"
    return f"shared://{first}"


def detect_peer_dependencies(text: str, service_name: str) -> set[str]:
    peers: set[str] = set()
    for host in EXPLICIT_SERVICE_PATTERN.findall(text):
        if host in SERVICE_FOLDERS and host != service_name:
            peers.add(host)
    for endpoint, peer in ENDPOINT_PEER_MAP.items():
        if endpoint in text and peer != service_name:
            peers.add(peer)
    return peers


def build_service_summary(
    repo_root: Path,
    compose_services: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    implementation_root = repo_root / "TGN" / "implementation"
    service_summaries: dict[str, dict[str, Any]] = {}

    for service_name, folder_name in SERVICE_FOLDERS.items():
        service_root = implementation_root / folder_name
        python_files = iter_python_files(service_root)
        combined_text = "\n".join(path.read_text() for path in python_files)
        imports = set().union(*(collect_imports(path) for path in python_files))
        peer_dependencies = detect_peer_dependencies(combined_text, service_name)
        shared_storage_dependencies = sorted(
            {
                normalize_shared_uri(match)
                for match in SHARED_URI_PATTERN.finditer(combined_text)
            }
        )

        shared_code_dependencies: list[str] = []
        if any(name == "service_registry" or name.startswith("service_registry.") for name in imports):
            shared_code_dependencies.append("service_registry.py")

        contract_name = service_name.removesuffix("-service") + ".openapi.yaml"
        contract_path = Path("contracts") / contract_name
        if (repo_root / contract_path).exists():
            shared_code_dependencies.append(str(contract_path))

        compose_service = compose_services.get(service_name, {})
        raw_env = compose_service.get("environment", {})
        if isinstance(raw_env, list):
            env_map = {}
            for item in raw_env:
                if isinstance(item, str) and "=" in item:
                    key, value = item.split("=", 1)
                    env_map[key] = value
            raw_env = env_map
        if not isinstance(raw_env, dict):
            raw_env = {}
        infrastructure_dependencies = [
            friendly_name
            for key, friendly_name in INFRA_ENV_KEYS.items()
            if key in raw_env and str(raw_env[key]).strip()
        ]

        service_summaries[service_name] = {
            "service": service_name,
            "outbound_peer_dependencies": sorted(peer_dependencies),
            "shared_code_dependencies": shared_code_dependencies,
            "shared_storage_dependencies": shared_storage_dependencies,
            "infrastructure_dependencies": infrastructure_dependencies,
        }

    for service_name, summary in service_summaries.items():
        inbound = sorted(
            other_name
            for other_name, other_summary in service_summaries.items()
            if service_name in other_summary["outbound_peer_dependencies"]
        )
        summary["inbound_peer_consumers"] = inbound
        summary["outbound_peer_count"] = len(summary["outbound_peer_dependencies"])
        summary["inbound_peer_count"] = len(inbound)

    adjacency_matrix = {
        source: {
            target: target in summary["outbound_peer_dependencies"]
            for target in SERVICE_FOLDERS
            if target != source
        }
        for source, summary in service_summaries.items()
    }

    max_outbound = max((summary["outbound_peer_count"] for summary in service_summaries.values()), default=0)
    max_inbound = max((summary["inbound_peer_count"] for summary in service_summaries.values()), default=0)

    return {
        "status": "complete",
        "services": [service_summaries[name] for name in SERVICE_FOLDERS],
        "adjacency_matrix": adjacency_matrix,
        "max_outbound_peer_count": max_outbound,
        "max_inbound_peer_count": max_inbound,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze TGN service coupling and dependency structure.")
    parser.add_argument("--output", default=None, help="Optional path for the JSON summary output.")
    args = parser.parse_args()

    repo_root = discover_repo_root()
    compose_services = load_docker_compose_services(repo_root / "TGN" / "implementation" / "docker-compose.yml")
    summary = build_service_summary(repo_root, compose_services)
    text = json.dumps(summary, indent=2) + "\n"

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)
        print(output_path)
    else:
        print(text, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
