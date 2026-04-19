#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_BIN = REPO_ROOT / "tools" / "bin"
DEFAULT_CLUSTER_NAME = "tsgv-scaling"
DEFAULT_NAMESPACE = "default"
DEFAULT_KIND_IMAGE = "kindest/node:v1.35.1"
DEFAULT_METRICS_SERVER_URL = (
    "https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.7.2/components.yaml"
)
KIND_STORAGE_PATH = "/mnt/tsgv-storage"
HPA_MIN_REPLICAS = 1
HPA_MAX_REPLICAS = 5
HPA_CPU_TARGET_PERCENT = 60
RESOURCE_REQUEST_CPU_M = 250
RESOURCE_REQUEST_MEMORY_MI = 512
RESOURCE_LIMIT_CPU_M = 1000
RESOURCE_LIMIT_MEMORY_MI = 1024

SERVICE_SPECS: dict[str, dict[str, Any]] = {
    "inference-service": {
        "port": 8005,
        "image": "tsgv-inference-service:scaling",
        "dockerfile": REPO_ROOT / "TGN" / "implementation" / "inference-service" / "Dockerfile",
        "endpoint": "/infer/ground",
        "fixture": REPO_ROOT / "kubernetes" / "scaling" / "fixtures" / "inference_request.json",
        "container_env": {
            "SERVICE_MODEL_NAME": "tgn",
            "SERVICE_IMPLEMENTATION_NAME": "tgn-inference-service-scaling",
        },
    },
    "text-processing-service": {
        "port": 8003,
        "image": "tsgv-text-processing-service:scaling",
        "dockerfile": REPO_ROOT / "TGN" / "implementation" / "text-processing-service" / "Dockerfile",
        "endpoint": "/text/batch",
        "fixture": REPO_ROOT / "kubernetes" / "scaling" / "fixtures" / "text_batch_request.json",
        "container_env": {
            "SERVICE_MODEL_NAME": "tgn",
            "SERVICE_IMPLEMENTATION_NAME": "tgn-text-processing-service-scaling",
            "TEXT_PREDICTION_SEMANTIC_ENABLED": "true",
        },
    },
}

REQUIRED_STORAGE_PATHS = [
    REPO_ROOT / "TGN" / "implementation" / "storage" / "models" / "manual-live-smoke.bin",
    REPO_ROOT / "TGN" / "implementation" / "storage" / "features" / "visual" / "s27-d34.vf.pt",
    REPO_ROOT / "TGN" / "implementation" / "storage" / "text" / "processed" / "manual-live-smoke-s27-d34-e1.tp.json",
    REPO_ROOT / "TGN" / "implementation" / "storage" / "artifacts" / "text" / "v1" / "vocab.json",
    REPO_ROOT / "TGN" / "implementation" / "storage" / "artifacts" / "text" / "v1" / "embeddings.npy",
    REPO_ROOT / "kubernetes" / "scaling" / "loadgen.py",
    REPO_ROOT / "kubernetes" / "scaling" / "fixtures" / "inference_request.json",
    REPO_ROOT / "kubernetes" / "scaling" / "fixtures" / "text_batch_request.json",
]


class CommandError(RuntimeError):
    pass


def command_line() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or defer kind-backed Kubernetes scaling experiments.")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["exp10_kubernetes_autoscaling_behavior", "exp11_horizontal_scaling_concurrent_requests"],
        help="Experiment folder name.",
    )
    parser.add_argument(
        "--services",
        default="inference-service,text-processing-service",
        help="Comma-separated logical services to test.",
    )
    parser.add_argument(
        "--service",
        action="append",
        default=[],
        help="Backward-compatible single-service selector. Can be passed multiple times.",
    )
    parser.add_argument(
        "--results-root",
        default="Results/thesis_experiments",
        help="Root directory that contains the thesis experiment folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Explicit experiment output directory. Defaults to <results-root>/<experiment>.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Backward-compatible path to summary.json; the bundle directory is derived from its parent.",
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="Kubernetes namespace.")
    parser.add_argument("--cluster-name", default=DEFAULT_CLUSTER_NAME, help="Kind cluster name.")
    parser.add_argument("--node-image", default=DEFAULT_KIND_IMAGE, help="Kind node image.")
    parser.add_argument(
        "--image-load-timeout-sec",
        type=float,
        default=180.0,
        help="Timeout in seconds for loading each Docker image into the kind node.",
    )
    parser.add_argument(
        "--metrics-server-manifest-url",
        default=DEFAULT_METRICS_SERVER_URL,
        help="Metrics-server manifest URL applied after cluster creation.",
    )
    parser.add_argument(
        "--kind-path",
        default=None,
        help="Optional explicit path to kind. Falls back to tools/bin/kind or PATH.",
    )
    parser.add_argument(
        "--kubectl-path",
        default=None,
        help="Optional explicit path to kubectl. Falls back to tools/bin/kubectl or PATH.",
    )
    parser.add_argument("--skip-build", action="store_true", help="Reuse existing local service images.")
    parser.add_argument("--keep-cluster", action="store_true", help="Leave the kind cluster running after the experiment.")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight and bundle generation without measurement.")
    parser.add_argument(
        "--update-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Regenerate Results/thesis_experiments/INDEX.md after a complete run.",
    )
    parser.add_argument(
        "--calibration-concurrency",
        default="5,10,20,40,80",
        help="Comma-separated constant-concurrency sweep for exp10 calibration.",
    )
    parser.add_argument(
        "--exp11-concurrency",
        default="1,5,10,20,40",
        help="Comma-separated concurrency sweep for exp11.",
    )
    parser.add_argument("--calibration-duration-sec", type=float, default=30.0)
    parser.add_argument("--exp10-warmup-sec", type=float, default=30.0)
    parser.add_argument("--exp10-hold-sec", type=float, default=240.0)
    parser.add_argument("--exp10-cooldown-sec", type=float, default=300.0)
    parser.add_argument("--exp11-stage-duration-sec", type=float, default=120.0)
    parser.add_argument("--exp11-warmup-sec", type=float, default=20.0)
    parser.add_argument("--exp11-baseline-cooldown-sec", type=float, default=30.0)
    parser.add_argument("--exp11-hpa-cooldown-sec", type=float, default=90.0)
    parser.add_argument("--replica-poll-sec", type=float, default=2.0)
    parser.add_argument("--pod-metrics-poll-sec", type=float, default=5.0)
    return parser.parse_args()


def resolve_tool(explicit: str | None, name: str) -> str:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{name} not found at {path}")
        return str(path)

    local = TOOLS_BIN / name
    if local.exists():
        return str(local)

    discovered = shutil.which(name)
    if discovered:
        return discovered

    raise FileNotFoundError(f"required tool not found: {name}")


def parse_csv_ints(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    parsed = [int(value) for value in values]
    if any(value <= 0 for value in parsed):
        raise ValueError("all concurrency values must be positive integers")
    return parsed


def selected_services(args: argparse.Namespace) -> list[str]:
    names: list[str] = []
    if args.service:
        names.extend(args.service)
    if args.services:
        names.extend(part.strip() for part in args.services.split(",") if part.strip())
    deduped = list(dict.fromkeys(names))
    if not deduped:
        deduped = list(SERVICE_SPECS)
    unknown = [name for name in deduped if name not in SERVICE_SPECS]
    if unknown:
        raise ValueError(f"unsupported services requested: {', '.join(unknown)}")
    return deduped


def run(
    command: list[str],
    *,
    capture: bool = True,
    check: bool = True,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=full_env,
        text=True,
        capture_output=capture,
        input=input_text,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise CommandError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def docker(
    *args: str,
    capture: bool = True,
    check: bool = True,
    cwd: Path | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return run(["docker", *args], capture=capture, check=check, cwd=cwd, timeout=timeout)


def kubectl(
    tool: str,
    *args: str,
    capture: bool = True,
    check: bool = True,
    input_text: str | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return run([tool, *args], capture=capture, check=check, input_text=input_text, timeout=timeout)


def kind(
    tool: str,
    *args: str,
    capture: bool = True,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return run([tool, *args], capture=capture, check=check, timeout=timeout)


def sanitize_name(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "scaling"


def parse_cpu_millicores(raw: str) -> int:
    value = raw.strip()
    if not value:
        return 0
    if value.endswith("m"):
        return int(float(value[:-1]))
    return int(float(value) * 1000.0)


def parse_memory_mib(raw: str) -> float:
    value = raw.strip()
    if not value:
        return 0.0
    suffixes = {
        "Ki": 1.0 / 1024.0,
        "Mi": 1.0,
        "Gi": 1024.0,
        "Ti": 1024.0 * 1024.0,
        "K": 1.0 / 1000.0,
        "M": 1000.0 / 1024.0,
        "G": (1000.0 * 1000.0) / (1024.0 * 1024.0),
    }
    for suffix, factor in suffixes.items():
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * factor
    return float(value) / float(1024 * 1024)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def output_dir_for(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    if args.output:
        return Path(args.output).resolve().parent
    return (REPO_ROOT / args.results_root / args.experiment).resolve()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def reset_bundle_dir(path: Path) -> tuple[Path, Path]:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    logs_dir = path / "logs"
    screenshots_dir = path / "screenshots"
    logs_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir, screenshots_dir


def ensure_required_artifacts() -> list[str]:
    return [str(path) for path in REQUIRED_STORAGE_PATHS if not path.exists()]


def docker_daemon_ok() -> tuple[bool, str]:
    try:
        result = docker("info", capture=True, check=False)
    except FileNotFoundError:
        return False, "docker was not found in PATH"
    if result.returncode == 0:
        return True, ""
    detail = (result.stderr or result.stdout).strip() or "docker info failed"
    return False, detail


def build_kind_config(storage_root: Path) -> str:
    return textwrap.dedent(
        f"""\
        kind: Cluster
        apiVersion: kind.x-k8s.io/v1alpha4
        nodes:
          - role: control-plane
            extraMounts:
              - hostPath: {storage_root}
                containerPath: {KIND_STORAGE_PATH}
        """
    )


def cluster_exists(kind_bin: str, cluster_name: str) -> bool:
    return cluster_name in kind(kind_bin, "get", "clusters").stdout.splitlines()


def ensure_python_image_available() -> None:
    inspection = docker("image", "inspect", "python:3.11-slim", capture=True, check=False)
    if inspection.returncode == 0:
        return
    docker("pull", "python:3.11-slim")


def kind_node_has_image(cluster_name: str, image: str) -> bool:
    node_name = f"{cluster_name}-control-plane"
    result = run(
        ["docker", "exec", node_name, "ctr", "-n=k8s.io", "images", "ls", "--quiet"],
        capture=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    available = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if image in available:
        return True
    repo, _, tag = image.partition(":")
    if tag:
        for candidate in available:
            if candidate == image:
                return True
            if candidate.startswith(f"{repo}:") and candidate.endswith(f":{tag}"):
                return True
            if candidate.startswith(f"docker.io/library/{repo}:") and candidate.endswith(tag):
                return True
            if candidate.startswith(f"docker.io/{repo}:") and candidate.endswith(tag):
                return True
    return False


def load_image_into_kind(kind_bin: str, cluster_name: str, image: str, timeout_sec: float) -> None:
    node_name = f"{cluster_name}-control-plane"
    if kind_node_has_image(cluster_name, image):
        return
    try:
        kind_result = kind(
            kind_bin,
            "load",
            "docker-image",
            image,
            "--name",
            cluster_name,
            capture=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        kind_result = subprocess.CompletedProcess(
            args=exc.cmd,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or f"kind load docker-image timed out after {timeout_sec} seconds",
        )
    if kind_result.returncode == 0:
        return

    with tempfile.NamedTemporaryFile(prefix="kind-image-", suffix=".tar", delete=False) as handle:
        archive_path = Path(handle.name)
    try:
        try:
            docker("save", image, "-o", str(archive_path), timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            raise CommandError(f"timed out saving image {image} after {timeout_sec} seconds") from exc
        with archive_path.open("rb") as handle:
            try:
                result = subprocess.run(
                    ["docker", "exec", "-i", node_name, "ctr", "-n=k8s.io", "images", "import", "-"],
                    stdin=handle,
                    text=False,
                    capture_output=True,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired as exc:
                raise CommandError(
                    f"timed out loading image {image} into kind after {timeout_sec} seconds\n"
                    f"kind stdout:\n{kind_result.stdout}\n"
                    f"kind stderr:\n{kind_result.stderr}"
                ) from exc
        if result.returncode != 0:
            raise CommandError(
                f"failed to load image {image} into kind\n"
                f"kind stdout:\n{kind_result.stdout}\n"
                f"kind stderr:\n{kind_result.stderr}\n"
                f"ctr stdout:\n{result.stdout.decode(errors='replace')}\n"
                f"ctr stderr:\n{result.stderr.decode(errors='replace')}"
            )
    finally:
        archive_path.unlink(missing_ok=True)


def create_cluster(
    kind_bin: str,
    cluster_name: str,
    node_image: str,
    storage_root: Path,
    logs_dir: Path,
    *,
    reuse_existing: bool,
) -> None:
    if cluster_exists(kind_bin, cluster_name):
        if reuse_existing:
            return
        kind(kind_bin, "delete", "cluster", "--name", cluster_name)

    config_text = build_kind_config(storage_root)
    write_text(logs_dir / "kind-config.yaml", config_text)
    kind(kind_bin, "create", "cluster", "--name", cluster_name, "--image", node_image, "--config", str(logs_dir / "kind-config.yaml"))


def delete_cluster(kind_bin: str, cluster_name: str) -> None:
    if cluster_exists(kind_bin, cluster_name):
        kind(kind_bin, "delete", "cluster", "--name", cluster_name)


def install_metrics_server(kubectl_bin: str, manifest_url: str) -> None:
    kubectl(kubectl_bin, "apply", "-f", manifest_url)
    patch = json.dumps(
        [
            {"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"},
            {
                "op": "add",
                "path": "/spec/template/spec/containers/0/args/-",
                "value": "--kubelet-preferred-address-types=InternalIP,Hostname,InternalDNS,ExternalDNS,ExternalIP",
            },
        ]
    )
    kubectl(
        kubectl_bin,
        "patch",
        "deployment",
        "metrics-server",
        "-n",
        "kube-system",
        "--type=json",
        "-p",
        patch,
        check=False,
    )
    kubectl(kubectl_bin, "rollout", "status", "deployment/metrics-server", "-n", "kube-system", "--timeout=240s")


def wait_for_pod_metrics(kubectl_bin: str, namespace: str, timeout_sec: float = 240.0) -> None:
    deadline = time.monotonic() + timeout_sec
    last_error = "kubectl top pods did not succeed"
    while time.monotonic() < deadline:
        result = kubectl(kubectl_bin, "top", "pods", "-n", namespace, capture=True, check=False)
        if result.returncode == 0:
            return
        last_error = (result.stderr or result.stdout).strip() or last_error
        time.sleep(5)
    raise CommandError(f"metrics-server or pod metrics were not ready: {last_error}")


def build_service_images(services: list[str], skip_build: bool) -> None:
    if skip_build:
        return
    for service_name in services:
        spec = SERVICE_SPECS[service_name]
        docker(
            "build",
            "-t",
            spec["image"],
            "-f",
            str(spec["dockerfile"]),
            ".",
            cwd=REPO_ROOT,
        )


def make_configmap(namespace: str) -> dict[str, Any]:
    files = {
        "loadgen.py": (REPO_ROOT / "kubernetes" / "scaling" / "loadgen.py").read_text(),
        "inference_request.json": (REPO_ROOT / "kubernetes" / "scaling" / "fixtures" / "inference_request.json").read_text(),
        "text_batch_request.json": (REPO_ROOT / "kubernetes" / "scaling" / "fixtures" / "text_batch_request.json").read_text(),
    }
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "scaling-config", "namespace": namespace},
        "data": files,
    }


def make_service_objects(service_name: str, namespace: str) -> list[dict[str, Any]]:
    spec = SERVICE_SPECS[service_name]
    port = int(spec["port"])
    service_url = f"http://{service_name}.{namespace}.svc.cluster.local:{port}"
    env = [
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "IMPLEMENTATION_STORAGE_ROOT", "value": "/app/storage"},
        {"name": "SERVICE_REGISTRY_ENABLED", "value": "false"},
        {"name": "SERVICE_RUNTIME_PLATFORM", "value": "kubernetes-kind"},
        {"name": "SERVICE_RUNTIME_NAMESPACE", "value": namespace},
        {"name": "SERVICE_SELF_URL", "value": service_url},
    ]
    for key, value in spec["container_env"].items():
        env.append({"name": key, "value": value})

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": service_name, "namespace": namespace},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": service_name}},
            "template": {
                "metadata": {"labels": {"app": service_name}},
                "spec": {
                    "containers": [
                        {
                            "name": "app",
                            "image": spec["image"],
                            "imagePullPolicy": "IfNotPresent",
                            "ports": [{"containerPort": port}],
                            "env": env,
                            "resources": {
                                "requests": {
                                    "cpu": f"{RESOURCE_REQUEST_CPU_M}m",
                                    "memory": f"{RESOURCE_REQUEST_MEMORY_MI}Mi",
                                },
                                "limits": {
                                    "cpu": f"{RESOURCE_LIMIT_CPU_M}m",
                                    "memory": f"{RESOURCE_LIMIT_MEMORY_MI}Mi",
                                },
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/registry/self", "port": port},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5,
                                "timeoutSeconds": 3,
                            },
                            "volumeMounts": [{"name": "storage", "mountPath": "/app/storage", "readOnly": True}],
                        }
                    ],
                    "volumes": [
                        {"name": "storage", "hostPath": {"path": KIND_STORAGE_PATH, "type": "Directory"}}
                    ],
                },
            },
        },
    }

    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": service_name, "namespace": namespace},
        "spec": {
            "selector": {"app": service_name},
            "ports": [{"name": "http", "port": port, "targetPort": port}],
        },
    }
    return [deployment, service]


def make_hpa(service_name: str, namespace: str) -> dict[str, Any]:
    return {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": f"{service_name}-hpa", "namespace": namespace},
        "spec": {
            "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": service_name},
            "minReplicas": HPA_MIN_REPLICAS,
            "maxReplicas": HPA_MAX_REPLICAS,
            "metrics": [
                {
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {"type": "Utilization", "averageUtilization": HPA_CPU_TARGET_PERCENT},
                    },
                }
            ],
        },
    }


def apply_objects(kubectl_bin: str, path: Path, objects: list[dict[str, Any]] | dict[str, Any]) -> None:
    payload = objects
    if isinstance(objects, list):
        payload = {"apiVersion": "v1", "kind": "List", "items": objects}
    write_json(path, payload)
    kubectl(kubectl_bin, "apply", "-f", str(path))


def delete_manifest(kubectl_bin: str, path: Path) -> None:
    if path.exists():
        kubectl(kubectl_bin, "delete", "-f", str(path), "--ignore-not-found=true", check=False)


def wait_for_deployment(kubectl_bin: str, namespace: str, name: str, timeout: str = "240s") -> None:
    kubectl(kubectl_bin, "rollout", "status", f"deployment/{name}", "-n", namespace, f"--timeout={timeout}")


def deployment_ready_replicas(kubectl_bin: str, namespace: str, name: str) -> int:
    payload = json.loads(kubectl(kubectl_bin, "get", "deployment", name, "-n", namespace, "-o", "json").stdout)
    status = payload.get("status", {})
    for key in ["availableReplicas", "readyReplicas", "replicas"]:
        value = status.get(key)
        if isinstance(value, int):
            return value
    return 0


def wait_for_single_replica(kubectl_bin: str, namespace: str, name: str, timeout_sec: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if deployment_ready_replicas(kubectl_bin, namespace, name) <= 1:
            return
        time.sleep(2)
    raise CommandError(f"deployment {name} did not settle to a single replica in {timeout_sec} seconds")


def sample_replica_timeline(
    kubectl_bin: str,
    namespace: str,
    deployment_name: str,
    interval_sec: float,
    stop_event: threading.Event,
    timeline: list[dict[str, Any]],
    start_time: float,
) -> None:
    while not stop_event.is_set():
        try:
            payload = json.loads(kubectl(kubectl_bin, "get", "deployment", deployment_name, "-n", namespace, "-o", "json").stdout)
            status = payload.get("status", {})
            timeline.append(
                {
                    "elapsed_sec": round(time.monotonic() - start_time, 3),
                    "replicas": int(status.get("replicas", 0) or 0),
                    "ready_replicas": int(status.get("readyReplicas", 0) or 0),
                    "available_replicas": int(status.get("availableReplicas", 0) or 0),
                }
            )
        except Exception:
            pass
        stop_event.wait(interval_sec)


def sample_pod_metrics(
    kubectl_bin: str,
    namespace: str,
    app_label: str,
    interval_sec: float,
    stop_event: threading.Event,
    samples: list[dict[str, Any]],
    start_time: float,
) -> None:
    while not stop_event.is_set():
        result = kubectl(
            kubectl_bin,
            "top",
            "pods",
            "-n",
            namespace,
            "-l",
            f"app={app_label}",
            "--no-headers",
            capture=True,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) < 3:
                    continue
                samples.append(
                    {
                        "elapsed_sec": round(time.monotonic() - start_time, 3),
                        "pod": parts[0],
                        "cpu_millicores": parse_cpu_millicores(parts[1]),
                        "memory_mib": round(parse_memory_mib(parts[2]), 3),
                    }
                )
        stop_event.wait(interval_sec)


def resource_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "average_cpu_millicores_per_pod": "not_run",
            "peak_cpu_millicores_per_pod": "not_run",
            "average_memory_mib_per_pod": "not_run",
            "peak_memory_mib_per_pod": "not_run",
        }
    cpus = [int(sample["cpu_millicores"]) for sample in samples]
    memories = [float(sample["memory_mib"]) for sample in samples]
    return {
        "average_cpu_millicores_per_pod": round(sum(cpus) / len(cpus), 3),
        "peak_cpu_millicores_per_pod": max(cpus),
        "average_memory_mib_per_pod": round(sum(memories) / len(memories), 3),
        "peak_memory_mib_per_pod": round(max(memories), 3),
    }


def peak_ready_replicas(timeline: list[dict[str, Any]]) -> int:
    return max((int(item.get("ready_replicas", 0)) for item in timeline), default=0)


def time_to_first_scale_up(timeline: list[dict[str, Any]]) -> float | str:
    for item in timeline:
        if int(item.get("ready_replicas", 0)) > 1:
            return float(item["elapsed_sec"])
    return "not_observed"


def time_to_stable_peak(timeline: list[dict[str, Any]]) -> float | str:
    peak = peak_ready_replicas(timeline)
    if peak <= 1:
        return "not_observed"
    streak = 0
    candidate = None
    for item in timeline:
        ready = int(item.get("ready_replicas", 0))
        if ready == peak:
            streak += 1
            if candidate is None:
                candidate = float(item["elapsed_sec"])
            if streak >= 3:
                return candidate
        else:
            streak = 0
            candidate = None
    return candidate if candidate is not None else "not_observed"


def cooldown_settle(peak: int, cooldown_timeline: list[dict[str, Any]]) -> tuple[bool, float | str]:
    if peak <= 1:
        return False, "not_observed"
    observed_drop = False
    settle_at: float | None = None
    for item in cooldown_timeline:
        ready = int(item.get("ready_replicas", 0))
        elapsed = float(item["elapsed_sec"])
        if ready < peak:
            observed_drop = True
        if observed_drop and ready <= 1:
            settle_at = elapsed
            break
    return observed_drop, (settle_at if settle_at is not None else "not_observed")


def make_load_job(
    job_name: str,
    namespace: str,
    service_name: str,
    payload_file: str,
    concurrency: int,
    duration_sec: float,
    warmup_sec: float,
) -> dict[str, Any]:
    spec = SERVICE_SPECS[service_name]
    url = f"http://{service_name}.{namespace}.svc.cluster.local:{spec['port']}{spec['endpoint']}"
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name, "namespace": namespace},
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {"labels": {"job-name": job_name}},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "loadgen",
                            "image": "python:3.11-slim",
                            "imagePullPolicy": "IfNotPresent",
                            "command": [
                                "python",
                                "/config/loadgen.py",
                                "--url",
                                url,
                                "--payload-file",
                                f"/config/{payload_file}",
                                "--concurrency",
                                str(concurrency),
                                "--duration-sec",
                                str(duration_sec),
                                "--warmup-sec",
                                str(warmup_sec),
                            ],
                            "volumeMounts": [{"name": "config", "mountPath": "/config", "readOnly": True}],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "128Mi"},
                                "limits": {"cpu": "500m", "memory": "256Mi"},
                            },
                        }
                    ],
                    "volumes": [{"name": "config", "configMap": {"name": "scaling-config"}}],
                },
            },
        },
    }


def job_pod_name(kubectl_bin: str, namespace: str, job_name: str) -> str:
    payload = json.loads(
        kubectl(
            kubectl_bin,
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"job-name={job_name}",
            "-o",
            "json",
        ).stdout
    )
    items = payload.get("items", [])
    if not items:
        raise CommandError(f"no pod found for job {job_name}")
    return str(items[0]["metadata"]["name"])


def run_load_stage(
    kubectl_bin: str,
    namespace: str,
    service_name: str,
    mode_label: str,
    stage_label: str,
    concurrency: int,
    duration_sec: float,
    warmup_sec: float,
    replica_poll_sec: float,
    pod_metrics_poll_sec: float,
    logs_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    payload_file = Path(SERVICE_SPECS[service_name]["fixture"]).name
    job_name = sanitize_name(
        f"lg-{service_name[:4]}-{mode_label}-{stage_label}-{int(time.time() * 1000) % 100000}"
    )[:63]
    job_path = logs_dir / f"{job_name}.job.json"
    apply_objects(
        kubectl_bin,
        job_path,
        make_load_job(
            job_name=job_name,
            namespace=namespace,
            service_name=service_name,
            payload_file=payload_file,
            concurrency=concurrency,
            duration_sec=duration_sec,
            warmup_sec=warmup_sec,
        ),
    )

    replica_timeline: list[dict[str, Any]] = []
    pod_metric_samples: list[dict[str, Any]] = []
    start_time = time.monotonic()
    stop_event = threading.Event()
    replica_thread = threading.Thread(
        target=sample_replica_timeline,
        args=(kubectl_bin, namespace, service_name, replica_poll_sec, stop_event, replica_timeline, start_time),
        daemon=True,
    )
    metric_thread = threading.Thread(
        target=sample_pod_metrics,
        args=(kubectl_bin, namespace, service_name, pod_metrics_poll_sec, stop_event, pod_metric_samples, start_time),
        daemon=True,
    )
    replica_thread.start()
    metric_thread.start()

    timeout_sec = max(60.0, duration_sec + 180.0)
    wait_result = kubectl(
        kubectl_bin,
        "wait",
        "--for=condition=complete",
        f"job/{job_name}",
        "-n",
        namespace,
        f"--timeout={int(timeout_sec)}s",
        capture=True,
        check=False,
    )

    stop_event.set()
    replica_thread.join(timeout=replica_poll_sec + 1.0)
    metric_thread.join(timeout=pod_metrics_poll_sec + 1.0)

    pod_name = job_pod_name(kubectl_bin, namespace, job_name)
    logs = kubectl(kubectl_bin, "logs", pod_name, "-n", namespace).stdout
    write_text(logs_dir / f"{job_name}.log", logs)
    write_json(logs_dir / f"{job_name}.replicas.json", replica_timeline)
    write_json(logs_dir / f"{job_name}.pod_metrics.json", pod_metric_samples)

    kubectl(kubectl_bin, "delete", "job", job_name, "-n", namespace, "--ignore-not-found=true", check=False)

    if wait_result.returncode != 0:
        raise CommandError(
            f"load job {job_name} did not complete successfully\n"
            f"stdout:\n{wait_result.stdout}\n"
            f"stderr:\n{wait_result.stderr}\n"
            f"logs:\n{logs}"
        )

    try:
        summary = json.loads(logs.strip())
    except json.JSONDecodeError as exc:
        raise CommandError(f"load job {job_name} did not emit valid JSON") from exc

    return summary, replica_timeline, pod_metric_samples


def run_validation_request(
    kubectl_bin: str,
    namespace: str,
    service_name: str,
    logs_dir: Path,
) -> dict[str, Any]:
    summary, _, _ = run_load_stage(
        kubectl_bin=kubectl_bin,
        namespace=namespace,
        service_name=service_name,
        mode_label="validate",
        stage_label="warmup",
        concurrency=1,
        duration_sec=3.0,
        warmup_sec=0.0,
        replica_poll_sec=2.0,
        pod_metrics_poll_sec=5.0,
        logs_dir=logs_dir,
    )
    if int(summary.get("success_count", 0)) <= 0 or int(summary.get("non_2xx_count", 0)) > 0:
        raise CommandError(f"warmup validation failed for {service_name}: {summary}")
    return summary


def capture_hpa_description(kubectl_bin: str, namespace: str, service_name: str) -> str:
    result = kubectl(
        kubectl_bin,
        "describe",
        "hpa",
        f"{service_name}-hpa",
        "-n",
        namespace,
        capture=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else ""


def deploy_service(
    kubectl_bin: str,
    namespace: str,
    service_name: str,
    logs_dir: Path,
) -> tuple[Path, Path]:
    service_path = logs_dir / f"{service_name}.resources.json"
    hpa_path = logs_dir / f"{service_name}.hpa.json"
    apply_objects(kubectl_bin, service_path, make_service_objects(service_name, namespace))
    wait_for_deployment(kubectl_bin, namespace, service_name)
    write_json(hpa_path, make_hpa(service_name, namespace))
    return service_path, hpa_path


def remove_service(kubectl_bin: str, service_path: Path, hpa_path: Path) -> None:
    delete_manifest(kubectl_bin, hpa_path)
    delete_manifest(kubectl_bin, service_path)


def apply_hpa(kubectl_bin: str, hpa_path: Path) -> None:
    kubectl(kubectl_bin, "apply", "-f", str(hpa_path))


def ensure_namespace(kubectl_bin: str, namespace: str) -> None:
    if namespace == "default":
        return
    fd, raw_path = tempfile.mkstemp(prefix="namespace-", suffix=".json")
    os.close(fd)
    path = Path(raw_path)
    try:
        apply_objects(
            kubectl_bin,
            path,
            {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {"name": namespace},
            },
        )
    finally:
        path.unlink(missing_ok=True)


def calibration_candidate(
    concurrency: int,
    load_summary: dict[str, Any],
    replica_timeline: list[dict[str, Any]],
    pod_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    resources = resource_summary(pod_metrics)
    return {
        "concurrency": concurrency,
        "requests_total": int(load_summary.get("measured_requests", 0)),
        "throughput_rps": float(load_summary.get("throughput_rps", 0.0)),
        "p50_latency_ms": float(load_summary.get("p50_latency_ms", 0.0)),
        "p95_latency_ms": float(load_summary.get("p95_latency_ms", 0.0)),
        "non_2xx_rate": round(
            int(load_summary.get("non_2xx_count", 0)) / max(1, int(load_summary.get("measured_requests", 0))),
            6,
        ),
        "replica_count_max": peak_ready_replicas(replica_timeline),
        **resources,
    }


def choose_calibrated_concurrency(candidates: list[dict[str, Any]], default_value: int) -> int:
    threshold_cpu = RESOURCE_REQUEST_CPU_M * 0.7
    for candidate in candidates:
        avg_cpu = candidate.get("average_cpu_millicores_per_pod")
        if isinstance(avg_cpu, (int, float)) and float(avg_cpu) >= threshold_cpu:
            return int(candidate["concurrency"])
    return default_value


def run_exp10_service(
    *,
    kubectl_bin: str,
    namespace: str,
    service_name: str,
    calibration_concurrency: list[int],
    calibration_duration_sec: float,
    warmup_sec: float,
    hold_sec: float,
    cooldown_sec: float,
    replica_poll_sec: float,
    pod_metrics_poll_sec: float,
    logs_dir: Path,
) -> dict[str, Any]:
    service_logs = logs_dir / service_name
    service_logs.mkdir(parents=True, exist_ok=True)
    service_path, hpa_path = deploy_service(kubectl_bin, namespace, service_name, service_logs)
    try:
        run_validation_request(kubectl_bin, namespace, service_name, service_logs)

        calibration: list[dict[str, Any]] = []
        for concurrency in calibration_concurrency:
            load_summary, replica_timeline, pod_metrics = run_load_stage(
                kubectl_bin=kubectl_bin,
                namespace=namespace,
                service_name=service_name,
                mode_label="cal",
                stage_label=str(concurrency),
                concurrency=concurrency,
                duration_sec=calibration_duration_sec,
                warmup_sec=0.0,
                replica_poll_sec=replica_poll_sec,
                pod_metrics_poll_sec=pod_metrics_poll_sec,
                logs_dir=service_logs,
            )
            calibration.append(calibration_candidate(concurrency, load_summary, replica_timeline, pod_metrics))

        selected_concurrency = choose_calibrated_concurrency(calibration, calibration_concurrency[-1])
        apply_hpa(kubectl_bin, hpa_path)
        hpa_before = capture_hpa_description(kubectl_bin, namespace, service_name)
        write_text(service_logs / f"{service_name}.hpa.before.txt", hpa_before)

        measurement_duration = warmup_sec + hold_sec
        load_summary, replica_timeline, pod_metrics = run_load_stage(
            kubectl_bin=kubectl_bin,
            namespace=namespace,
            service_name=service_name,
            mode_label="hpa",
            stage_label="measure",
            concurrency=selected_concurrency,
            duration_sec=measurement_duration,
            warmup_sec=warmup_sec,
            replica_poll_sec=replica_poll_sec,
            pod_metrics_poll_sec=pod_metrics_poll_sec,
            logs_dir=service_logs,
        )

        cooldown_timeline: list[dict[str, Any]] = []
        cooldown_stop = threading.Event()
        cooldown_thread = threading.Thread(
            target=sample_replica_timeline,
            args=(kubectl_bin, namespace, service_name, replica_poll_sec, cooldown_stop, cooldown_timeline, time.monotonic()),
            daemon=True,
        )
        cooldown_thread.start()
        time.sleep(max(0.0, cooldown_sec))
        cooldown_stop.set()
        cooldown_thread.join(timeout=replica_poll_sec + 1.0)
        write_json(service_logs / f"{service_name}.cooldown.replicas.json", cooldown_timeline)

        combined_timeline = replica_timeline + [
            {**item, "elapsed_sec": round(float(item["elapsed_sec"]) + measurement_duration, 3)}
            for item in cooldown_timeline
        ]
        peak = peak_ready_replicas(combined_timeline)
        scale_down_observed, scale_down_settle_sec = cooldown_settle(peak, cooldown_timeline)
        hpa_after = capture_hpa_description(kubectl_bin, namespace, service_name)
        write_text(service_logs / f"{service_name}.hpa.after.txt", hpa_after)
        run_validation_request(kubectl_bin, namespace, service_name, service_logs)

        return {
            "service": service_name,
            "status": "complete",
            "warmup_request_succeeded": True,
            "hpa": {
                "min_replicas": HPA_MIN_REPLICAS,
                "max_replicas": HPA_MAX_REPLICAS,
                "cpu_target_utilization_percent": HPA_CPU_TARGET_PERCENT,
            },
            "calibration": {
                "candidates": calibration,
                "selected_concurrency": selected_concurrency,
            },
            "load_profile": {
                "warmup_sec": warmup_sec,
                "hold_sec": hold_sec,
                "cooldown_sec": cooldown_sec,
                "selected_concurrency": selected_concurrency,
            },
            "load_result": {
                "requests_total": int(load_summary.get("measured_requests", 0)),
                "throughput_rps": float(load_summary.get("throughput_rps", 0.0)),
                "p50_latency_ms": float(load_summary.get("p50_latency_ms", 0.0)),
                "p95_latency_ms": float(load_summary.get("p95_latency_ms", 0.0)),
                "non_2xx_rate": round(
                    int(load_summary.get("non_2xx_count", 0)) / max(1, int(load_summary.get("measured_requests", 0))),
                    6,
                ),
            },
            "time_to_first_scale_up_sec": time_to_first_scale_up(combined_timeline),
            "time_to_stable_max_replicas_sec": time_to_stable_peak(combined_timeline),
            "peak_replicas": peak,
            "scale_down_observed": scale_down_observed,
            "scale_down_settle_sec": scale_down_settle_sec,
            "pod_resource_stats": resource_summary(pod_metrics),
            "replica_timeline": combined_timeline,
        }
    finally:
        remove_service(kubectl_bin, service_path, hpa_path)


def run_exp11_service(
    *,
    kubectl_bin: str,
    namespace: str,
    service_name: str,
    concurrency_values: list[int],
    stage_duration_sec: float,
    warmup_sec: float,
    baseline_cooldown_sec: float,
    hpa_cooldown_sec: float,
    replica_poll_sec: float,
    pod_metrics_poll_sec: float,
    logs_dir: Path,
) -> dict[str, Any]:
    service_logs = logs_dir / service_name
    service_logs.mkdir(parents=True, exist_ok=True)
    service_path, hpa_path = deploy_service(kubectl_bin, namespace, service_name, service_logs)
    try:
        run_validation_request(kubectl_bin, namespace, service_name, service_logs)
        result = {
            "service": service_name,
            "status": "complete",
            "baseline_single_replica": {"stage_results": []},
            "hpa_enabled": {"stage_results": []},
        }

        for mode in ["baseline_single_replica", "hpa_enabled"]:
            if mode == "hpa_enabled":
                apply_hpa(kubectl_bin, hpa_path)
                write_text(service_logs / f"{service_name}.{mode}.hpa.before.txt", capture_hpa_description(kubectl_bin, namespace, service_name))
            else:
                delete_manifest(kubectl_bin, hpa_path)

            kubectl(kubectl_bin, "scale", "deployment", service_name, "-n", namespace, "--replicas=1")
            wait_for_single_replica(kubectl_bin, namespace, service_name)
            run_validation_request(kubectl_bin, namespace, service_name, service_logs)

            stage_results: list[dict[str, Any]] = []
            for concurrency in concurrency_values:
                load_summary, replica_timeline, pod_metrics = run_load_stage(
                    kubectl_bin=kubectl_bin,
                    namespace=namespace,
                    service_name=service_name,
                    mode_label="base" if mode == "baseline_single_replica" else "hpa",
                    stage_label=str(concurrency),
                    concurrency=concurrency,
                    duration_sec=stage_duration_sec,
                    warmup_sec=warmup_sec,
                    replica_poll_sec=replica_poll_sec,
                    pod_metrics_poll_sec=pod_metrics_poll_sec,
                    logs_dir=service_logs,
                )
                stage_results.append(
                    {
                        "stage_duration_sec": stage_duration_sec,
                        "warmup_sec": warmup_sec,
                        "concurrency": concurrency,
                        "requests_total": int(load_summary.get("measured_requests", 0)),
                        "raw_requests_total": int(load_summary.get("requests_total", 0)),
                        "throughput_rps": float(load_summary.get("throughput_rps", 0.0)),
                        "p50_latency_ms": float(load_summary.get("p50_latency_ms", 0.0)),
                        "p95_latency_ms": float(load_summary.get("p95_latency_ms", 0.0)),
                        "non_2xx_rate": round(
                            int(load_summary.get("non_2xx_count", 0)) / max(1, int(load_summary.get("measured_requests", 0))),
                            6,
                        ),
                        "replica_count_max": peak_ready_replicas(replica_timeline),
                        **resource_summary(pod_metrics),
                    }
                )

                cooldown = baseline_cooldown_sec if mode == "baseline_single_replica" else hpa_cooldown_sec
                time.sleep(max(0.0, cooldown))
                wait_for_single_replica(kubectl_bin, namespace, service_name)

            result[mode]["stage_results"] = stage_results
            if mode == "hpa_enabled":
                write_text(service_logs / f"{service_name}.{mode}.hpa.after.txt", capture_hpa_description(kubectl_bin, namespace, service_name))

        run_validation_request(kubectl_bin, namespace, service_name, service_logs)
        return result
    finally:
        remove_service(kubectl_bin, service_path, hpa_path)


def preflight_summary(args: argparse.Namespace, services: list[str], kind_bin: str, kubectl_bin: str) -> dict[str, Any]:
    docker_ok, docker_detail = docker_daemon_ok()
    missing_paths = ensure_required_artifacts()
    return {
        "requested_services": services,
        "kind_bin": kind_bin,
        "kubectl_bin": kubectl_bin,
        "docker_ok": docker_ok,
        "docker_detail": docker_detail,
        "missing_artifacts": missing_paths,
        "cluster_name": args.cluster_name,
        "namespace": args.namespace,
    }


def experiment_summary(
    experiment: str,
    status: str,
    reason: str,
    services: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary = {
        "status": status,
        "cluster_provider": "kind",
        "cluster_name": args.cluster_name,
        "namespace": args.namespace,
        "services": services,
    }
    if reason:
        summary["reason"] = reason
    if experiment == "exp10_kubernetes_autoscaling_behavior":
        summary["hpa_policy"] = {
            "min_replicas": HPA_MIN_REPLICAS,
            "max_replicas": HPA_MAX_REPLICAS,
            "cpu_target_utilization_percent": HPA_CPU_TARGET_PERCENT,
        }
    return summary


def render_readme(args: argparse.Namespace, services: list[str]) -> str:
    return (
        f"Run context: {datetime.now(timezone.utc).isoformat()} on host `{socket.gethostname()}` "
        f"(`{platform.platform()}`) using a disposable `kind` cluster named `{args.cluster_name}` in namespace "
        f"`{args.namespace}`. `--keep-cluster` was {'enabled' if args.keep_cluster else 'disabled'}. The measured services were `{', '.join(services)}`, and the cluster-mounted storage root "
        f"was `{REPO_ROOT / 'TGN' / 'implementation' / 'storage'}`."
    )


def render_writeup(experiment: str, summary: dict[str, Any]) -> str:
    status = str(summary.get("status", "partial"))
    if status != "complete":
        reason = str(summary.get("reason", "The experiment did not complete.")).strip()
        if experiment == "exp10_kubernetes_autoscaling_behavior":
            return (
                f"The Kubernetes autoscaling experiment did not complete on this host. {reason}\n\n"
                "No autoscaling measurement was claimed beyond the fields captured in `summary.json`."
            )
        return (
            f"The concurrent horizontal-scaling experiment did not complete on this host. {reason}\n\n"
            "No scaling claim was made beyond the fields captured in `summary.json`."
        )

    services = summary.get("services", [])
    if experiment == "exp10_kubernetes_autoscaling_behavior":
        first = []
        second = []
        for item in services:
            first.append(
                f"`{item['service']}` scaled to `{item['peak_replicas']}` replicas with first scale-up observed after "
                f"`{item['time_to_first_scale_up_sec']}` seconds"
            )
            second.append(
                f"`{item['service']}` used calibrated concurrency `{item['calibration']['selected_concurrency']}`"
            )
        return (
            "The autoscaling experiment was run on a local `kind` cluster with HPA configured for CPU target utilization "
            f"`{HPA_CPU_TARGET_PERCENT}%`. " + "; ".join(first) + ".\n\n" +
            "Calibration and per-pod resource measurements were captured for both services. " + "; ".join(second) + "."
        )

    lines = []
    for item in services:
        baseline = item["baseline_single_replica"]["stage_results"]
        hpa = item["hpa_enabled"]["stage_results"]
        if baseline and hpa:
            lines.append(
                f"`{item['service']}` recorded baseline throughput `{baseline[-1]['throughput_rps']}` rps and "
                f"HPA-enabled throughput `{hpa[-1]['throughput_rps']}` rps at concurrency `{hpa[-1]['concurrency']}`"
            )
    return (
        "The concurrent horizontal-scaling experiment compared single-replica and HPA-enabled runs across the fixed "
        "concurrency sweep `1, 5, 10, 20, 40`. " + "; ".join(lines) + ".\n\n"
        "Per-stage latency, throughput, non-2xx rate, replica maxima, and pod resource measurements were captured in "
        "`summary.json`."
    )


def write_bundle(
    bundle_dir: Path,
    summary: dict[str, Any],
    args: argparse.Namespace,
    services: list[str],
) -> None:
    logs_dir = bundle_dir / "logs"
    screenshots_dir = bundle_dir / "screenshots"
    logs_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    write_json(bundle_dir / "summary.json", summary)
    write_text(bundle_dir / "writeup.md", render_writeup(args.experiment, summary) + "\n")
    write_text(bundle_dir / "command.txt", command_line() + "\n")
    write_text(bundle_dir / "README.md", render_readme(args, services) + "\n")
    write_json(logs_dir / "summary_excerpt.json", summary)
    if str(summary.get("status")) != "complete":
        write_text(logs_dir / "deferred_or_partial_reason.log", str(summary.get("reason", "")).strip() + "\n")


def maybe_update_index(results_root: Path) -> None:
    run(
        [sys.executable, str(REPO_ROOT / "scripts" / "build_thesis_synthesis.py"), "--results-root", str(results_root)],
        capture=True,
        check=True,
    )


def main() -> int:
    args = parse_args()
    services = selected_services(args)
    bundle_dir = output_dir_for(args)

    logs_dir, _ = reset_bundle_dir(bundle_dir)
    write_text(bundle_dir / "command.txt", command_line() + "\n")
    write_text(bundle_dir / "README.md", render_readme(args, services) + "\n")

    try:
        kind_bin = resolve_tool(args.kind_path, "kind")
        kubectl_bin = resolve_tool(args.kubectl_path, "kubectl")
    except FileNotFoundError as exc:
        summary = experiment_summary(args.experiment, "deferred", str(exc), [], args)
        write_bundle(bundle_dir, summary, args, services)
        print(bundle_dir)
        return 0

    preflight = preflight_summary(args, services, kind_bin, kubectl_bin)
    write_json(logs_dir / "preflight.json", preflight)

    reason_parts: list[str] = []
    if not preflight["docker_ok"]:
        reason_parts.append(preflight["docker_detail"])
    if preflight["missing_artifacts"]:
        reason_parts.append("missing required artifacts: " + ", ".join(preflight["missing_artifacts"]))
    if args.preflight_only:
        reason_parts.append("preflight_only was requested, so measurement was skipped.")

    if reason_parts:
        summary = experiment_summary(args.experiment, "deferred", " ".join(reason_parts), [], args)
        write_bundle(bundle_dir, summary, args, services)
        print(bundle_dir)
        return 0

    cluster_created = False
    service_results: list[dict[str, Any]] = []
    status = "complete"
    reason = ""

    try:
        create_cluster(
            kind_bin,
            args.cluster_name,
            args.node_image,
            REPO_ROOT / "TGN" / "implementation" / "storage",
            logs_dir,
            reuse_existing=args.keep_cluster,
        )
        cluster_created = True
        ensure_python_image_available()
        build_service_images(services, args.skip_build)
        for image in [SERVICE_SPECS[name]["image"] for name in services] + ["python:3.11-slim"]:
            load_image_into_kind(kind_bin, args.cluster_name, image, args.image_load_timeout_sec)

        install_metrics_server(kubectl_bin, args.metrics_server_manifest_url)
        ensure_namespace(kubectl_bin, args.namespace)
        wait_for_pod_metrics(kubectl_bin, args.namespace)
        apply_objects(kubectl_bin, logs_dir / "scaling-config.json", make_configmap(args.namespace))

        if args.experiment == "exp10_kubernetes_autoscaling_behavior":
            calibration_values = parse_csv_ints(args.calibration_concurrency)
            for service_name in services:
                service_results.append(
                    run_exp10_service(
                        kubectl_bin=kubectl_bin,
                        namespace=args.namespace,
                        service_name=service_name,
                        calibration_concurrency=calibration_values,
                        calibration_duration_sec=args.calibration_duration_sec,
                        warmup_sec=args.exp10_warmup_sec,
                        hold_sec=args.exp10_hold_sec,
                        cooldown_sec=args.exp10_cooldown_sec,
                        replica_poll_sec=args.replica_poll_sec,
                        pod_metrics_poll_sec=args.pod_metrics_poll_sec,
                        logs_dir=logs_dir,
                    )
                )
        else:
            concurrency_values = parse_csv_ints(args.exp11_concurrency)
            for service_name in services:
                service_results.append(
                    run_exp11_service(
                        kubectl_bin=kubectl_bin,
                        namespace=args.namespace,
                        service_name=service_name,
                        concurrency_values=concurrency_values,
                        stage_duration_sec=args.exp11_stage_duration_sec,
                        warmup_sec=args.exp11_warmup_sec,
                        baseline_cooldown_sec=args.exp11_baseline_cooldown_sec,
                        hpa_cooldown_sec=args.exp11_hpa_cooldown_sec,
                        replica_poll_sec=args.replica_poll_sec,
                        pod_metrics_poll_sec=args.pod_metrics_poll_sec,
                        logs_dir=logs_dir,
                    )
                )
    except Exception as exc:
        reason = str(exc).strip() or exc.__class__.__name__
        status = "partial" if service_results else "deferred"
    finally:
        if cluster_created and not args.keep_cluster:
            delete_cluster(kind_bin, args.cluster_name)

    summary = experiment_summary(args.experiment, status, reason, service_results, args)
    write_bundle(bundle_dir, summary, args, services)
    if status == "complete" and args.update_index:
        maybe_update_index((REPO_ROOT / args.results_root).resolve())
    print(bundle_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
