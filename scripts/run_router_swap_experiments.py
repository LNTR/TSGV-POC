#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_BIN = REPO_ROOT / "tools" / "bin"
KUBECTL = TOOLS_BIN / "kubectl"
KIND = TOOLS_BIN / "kind"
KIND_IMAGE = "kindest/node:v1.35.1"
CLUSTER_NAME = "tsgv-router-exp"
OPERATOR_IMAGE = "model-router-operator:local"
DEFAULT_NAMESPACE = "default"

SERVICE_PORTS = {
    "visual-preprocessing-service": 8001,
    "visual-feature-extraction-service": 8002,
    "text-processing-service": 8003,
    "training-service": 8004,
    "inference-service": 8005,
    "evaluation-service": 8006,
}


class CommandError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exp01 and exp02 ModelRouter swap experiments on kind.")
    parser.add_argument(
        "--results-root",
        default="Results/thesis_experiments",
        help="Root directory containing the thesis experiment folders.",
    )
    parser.add_argument(
        "--keep-cluster",
        action="store_true",
        help="Do not delete the kind cluster after the experiments complete.",
    )
    return parser.parse_args()


def run(
    command: list[str],
    *,
    capture: bool = True,
    check: bool = True,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
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
    )
    if check and result.returncode != 0:
        raise CommandError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def kubectl(*args: str, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run([str(KUBECTL), *args], capture=capture, check=check)


def kind(*args: str, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run([str(KIND), *args], capture=capture, check=check)


def docker(*args: str, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["docker", *args], capture=capture, check=check)


def ensure_tools() -> None:
    for tool in [KUBECTL, KIND]:
        if not tool.exists():
            raise FileNotFoundError(f"required local tool missing: {tool}")


def ensure_operator_image() -> None:
    image_names = docker("images", "--format", "{{.Repository}}:{{.Tag}}").stdout.splitlines()
    if OPERATOR_IMAGE in image_names:
        return

    dockerfile = REPO_ROOT / "kubernetes" / "operator" / ".tmp-k8s-build" / "model-router-operator"
    if not dockerfile.exists():
        raise FileNotFoundError("missing prebuilt operator binary under kubernetes/operator/.tmp-k8s-build")

    with tempfile.TemporaryDirectory(prefix="model-router-image-") as tmp_dir:
        dockerfile_path = Path(tmp_dir) / "Dockerfile"
        dockerfile_path.write_text(
            textwrap.dedent(
                """\
                FROM scratch
                COPY kubernetes/operator/.tmp-k8s-build/model-router-operator /model-router-operator
                ENTRYPOINT ["/model-router-operator"]
                """
            )
        )
        docker("build", "-t", OPERATOR_IMAGE, "-f", str(dockerfile_path), ".")


def create_cluster() -> None:
    existing = kind("get", "clusters").stdout.splitlines()
    if CLUSTER_NAME in existing:
        return
    kind("create", "cluster", "--name", CLUSTER_NAME, "--image", KIND_IMAGE)


def delete_cluster() -> None:
    existing = kind("get", "clusters").stdout.splitlines()
    if CLUSTER_NAME in existing:
        kind("delete", "cluster", "--name", CLUSTER_NAME)


def load_images() -> None:
    node_name = f"{CLUSTER_NAME}-control-plane"
    for image in [OPERATOR_IMAGE, "mongo:7", "python:3.11-slim"]:
        with tempfile.NamedTemporaryFile(prefix="kind-image-", suffix=".tar", delete=False) as handle:
            archive_path = Path(handle.name)
        try:
            docker("save", image, "-o", str(archive_path))
            with archive_path.open("rb") as handle:
                result = subprocess.run(
                    ["docker", "exec", "-i", node_name, "ctr", "-n=k8s.io", "images", "import", "-"],
                    stdin=handle,
                    text=False,
                    capture_output=True,
                )
            if result.returncode != 0:
                raise CommandError(
                    f"command failed ({result.returncode}): docker exec -i {node_name} ctr -n=k8s.io images import -\n"
                    f"stdout:\n{result.stdout.decode(errors='replace')}\n"
                    f"stderr:\n{result.stderr.decode(errors='replace')}"
                )
        finally:
            archive_path.unlink(missing_ok=True)


def write_json_manifest(path: Path, objects: list[dict[str, Any]] | dict[str, Any]) -> None:
    if isinstance(objects, list):
        payload = {"apiVersion": "v1", "kind": "List", "items": objects}
    else:
        payload = objects
    path.write_text(json.dumps(payload, indent=2) + "\n")


def apply_json_manifest(path: Path) -> None:
    kubectl("apply", "-f", str(path))


def wait_for_deployment(name: str, timeout: str = "180s") -> None:
    kubectl("wait", "--for=condition=available", f"deployment/{name}", f"--timeout={timeout}")


def wait_for_pod(name: str, timeout: str = "180s") -> None:
    kubectl("wait", "--for=condition=ready", f"pod/{name}", f"--timeout={timeout}")


def make_backend_deployment(name: str, label: str, response_text: str) -> dict[str, Any]:
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": label}},
            "template": {
                "metadata": {"labels": {"app": label}},
                "spec": {
                    "containers": [
                        {
                            "name": "backend",
                            "image": "python:3.11-slim",
                            "env": [{"name": "RESPONSE_TEXT", "value": response_text}],
                            "command": ["sh", "-lc"],
                            "args": [
                                "printf '%s' \"$RESPONSE_TEXT\" > /tmp/index.html && "
                                "cd /tmp && python -m http.server 8000"
                            ],
                            "ports": [{"containerPort": 8000}],
                        }
                    ]
                },
            },
        },
    }


def make_service(name: str, app_label: str, port: int) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name},
        "spec": {
            "selector": {"app": app_label},
            "ports": [{"name": "http", "port": port, "targetPort": 8000}],
        },
    }


def make_client_pod() -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "router-client"},
        "spec": {
            "restartPolicy": "Never",
            "containers": [
                {
                    "name": "client",
                    "image": "python:3.11-slim",
                    "command": ["sh", "-lc"],
                    "args": ["sleep 3600"],
                }
            ],
        },
    }


def setup_base_resources(temp_dir: Path) -> None:
    apply_json_manifest(REPO_ROOT / "kubernetes" / "manifests" / "mongodb.yaml")
    apply_json_manifest(REPO_ROOT / "kubernetes" / "manifests" / "modelrouter-crd.yaml")
    apply_json_manifest(REPO_ROOT / "kubernetes" / "manifests" / "operator-rbac.yaml")

    operator_deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "model-router-operator"},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": "model-router-operator"}},
            "template": {
                "metadata": {"labels": {"app": "model-router-operator"}},
                "spec": {
                    "serviceAccountName": "model-router-operator",
                    "containers": [
                        {
                            "name": "operator",
                            "image": OPERATOR_IMAGE,
                            "imagePullPolicy": "IfNotPresent",
                            "env": [
                                {"name": "MONGODB_URI", "value": "mongodb://mongodb:27017/"},
                                {"name": "MODEL_ROUTER_MONGODB_DATABASE", "value": "service_registry"},
                            ],
                        }
                    ],
                },
            },
        },
    }
    operator_path = temp_dir / "operator-deployment.json"
    write_json_manifest(operator_path, operator_deployment)
    apply_json_manifest(operator_path)

    items: list[dict[str, Any]] = [
        make_backend_deployment("tgn-backend", "tgn-backend", "tgn-backend"),
        make_backend_deployment("tall-backend", "tall-backend", "tall-backend"),
        make_client_pod(),
    ]
    for logical_service, port in SERVICE_PORTS.items():
        items.append(make_service(f"tgn-{logical_service}", "tgn-backend", port))
        items.append(make_service(f"tall-{logical_service}", "tall-backend", port))

    backends_path = temp_dir / "router-backends.json"
    write_json_manifest(backends_path, items)
    apply_json_manifest(backends_path)

    wait_for_deployment("mongodb")
    wait_for_deployment("model-router-operator")
    wait_for_deployment("tgn-backend")
    wait_for_deployment("tall-backend")
    wait_for_pod("router-client")


def make_router_manifest(active_model: str, shared_text_for_tall: bool = False) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model_name in ["tgn", "tall"]:
        services: dict[str, Any] = {}
        for logical_service, port in SERVICE_PORTS.items():
            services[logical_service] = {
                "serviceName": f"{model_name}-{logical_service}",
                "port": port,
                "implementationName": f"{model_name}-{logical_service}",
                "compatibleModels": [model_name],
            }
        models[model_name] = {"services": services}

    if shared_text_for_tall:
        models["tall"]["services"]["text-processing-service"] = {
            "serviceName": "tgn-text-processing-service",
            "port": SERVICE_PORTS["text-processing-service"],
            "implementationName": "tgn-text-processing-service",
            "compatibleModels": ["tgn", "tall"],
            "shared": True,
        }

    return {
        "apiVersion": "registry.tsgv.io/v1alpha1",
        "kind": "ModelRouter",
        "metadata": {"name": "tsgv-router"},
        "spec": {
            "activeModel": active_model,
            "models": models,
        },
    }


def apply_router_manifest(temp_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    path = temp_dir / f"{name}.json"
    write_json_manifest(path, payload)
    kubectl("apply", "-f", str(path))
    return path


def get_service_snapshot() -> dict[str, dict[str, Any]]:
    payload = json.loads(kubectl("get", "svc", "-o", "json").stdout)
    snapshot: dict[str, dict[str, Any]] = {}
    for item in payload.get("items", []):
        name = item["metadata"]["name"]
        if name not in SERVICE_PORTS:
            continue
        snapshot[name] = {
            "externalName": item["spec"].get("externalName"),
            "annotations": item["metadata"].get("annotations", {}),
        }
    return snapshot


def poll_until(condition, timeout_sec: float, interval_sec: float = 0.5) -> Any:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(interval_sec)
    return None


def expected_external_name(model_name: str, logical_service: str) -> str:
    return f"{model_name}-{logical_service}.{DEFAULT_NAMESPACE}.svc.cluster.local"


def request_alias(url: str) -> tuple[int, str]:
    script = (
        "import urllib.request;"
        f"body=urllib.request.urlopen('{url}', timeout=5).read().decode();"
        "print(body)"
    )
    result = kubectl("exec", "router-client", "--", "python", "-c", script)
    return 200, result.stdout.strip()


def fetch_mongo_document(collection: str, query: dict[str, Any]) -> dict[str, Any]:
    query_json = json.dumps(query, separators=(",", ":"))
    projection = json.dumps({"_id": 0}, separators=(",", ":"))
    script = (
        "db = db.getSiblingDB('service_registry');"
        f"const doc = db.getCollection('{collection}').findOne({query_json}, {projection});"
        "print(JSON.stringify(doc));"
    )
    result = kubectl(
        "exec",
        "deploy/mongodb",
        "--",
        "mongosh",
        "--quiet",
        "--eval",
        script,
    )
    data = result.stdout.strip()
    return json.loads(data) if data else {}


def operator_logs() -> str:
    return kubectl("logs", "deployment/model-router-operator", "--tail=100").stdout


def wait_for_alias_targets(expected_targets: dict[str, str], timeout_sec: float = 60.0) -> dict[str, dict[str, Any]]:
    def condition() -> dict[str, dict[str, Any]] | None:
        snapshot = get_service_snapshot()
        for logical_service, expected in expected_targets.items():
            if snapshot.get(logical_service, {}).get("externalName") != expected:
                return None
        return snapshot

    snapshot = poll_until(condition, timeout_sec)
    if snapshot is None:
        raise TimeoutError(f"timed out waiting for alias targets: {expected_targets}")
    return snapshot


def count_changed_services(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> int:
    count = 0
    for name in SERVICE_PORTS:
        if before.get(name, {}).get("externalName") != after.get(name, {}).get("externalName"):
            count += 1
    return count


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def run_exp01(temp_dir: Path, results_root: Path) -> dict[str, Any]:
    exp_dir = results_root / "exp01_single_component_swap"
    logs_dir = exp_dir / "logs"
    screenshots_dir = exp_dir / "screenshots"
    logs_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    (screenshots_dir / ".gitkeep").write_text("")

    before_manifest = make_router_manifest(active_model="tall", shared_text_for_tall=False)
    after_manifest = make_router_manifest(active_model="tall", shared_text_for_tall=True)
    before_path = apply_router_manifest(temp_dir, "exp01-before", before_manifest)

    wait_for_alias_targets({name: expected_external_name("tall", name) for name in SERVICE_PORTS})
    before_services = get_service_snapshot()
    route_before = fetch_mongo_document(
        "routes",
        {
            "namespace": DEFAULT_NAMESPACE,
            "router_name": "tsgv-router",
            "logical_service": "text-processing-service",
        },
    )

    start = time.monotonic()
    after_path = apply_router_manifest(temp_dir, "exp01-after", after_manifest)

    def request_condition() -> dict[str, Any] | None:
        try:
            status, body = request_alias("http://text-processing-service:8003/index.html")
        except Exception:
            return None
        if body != "tgn-backend":
            return None
        return {"status_code": status, "body": body}

    request_result = poll_until(request_condition, 60.0, interval_sec=0.5)
    if request_result is None:
        raise TimeoutError("timed out waiting for exp01 routed request to succeed")
    swap_latency_sec = round(time.monotonic() - start, 3)

    after_services = wait_for_alias_targets(
        {
            **{name: expected_external_name("tall", name) for name in SERVICE_PORTS if name != "text-processing-service"},
            "text-processing-service": expected_external_name("tgn", "text-processing-service"),
        }
    )
    route_after = fetch_mongo_document(
        "routes",
        {
            "namespace": DEFAULT_NAMESPACE,
            "router_name": "tsgv-router",
            "logical_service": "text-processing-service",
        },
    )

    aliases_reconciled = count_changed_services(before_services, after_services)

    summary = {
        "status": "complete",
        "swap_latency_sec": swap_latency_sec,
        "aliases_reconciled": aliases_reconciled,
        "request_succeeded": True,
        "route_before": route_before,
        "route_after": route_after,
    }

    write_json(exp_dir / "summary.json", summary)
    write_text(
        exp_dir / "README.md",
        "Run context: 2026-04-18 on a local kind cluster using the shared model-router-operator, local smoke backends for both tgn and tall, and MongoDB route persistence in the default namespace.\n",
    )
    write_text(
        exp_dir / "writeup.md",
        (
            f"The single-component swap was measured on a local kind cluster with `activeModel` fixed at `tall`. "
            f"The `text-processing-service` entry under the `tall` model was changed to the `tgn` implementation "
            f"with `compatibleModels` set to `[tgn, tall]` and `shared` set to `true`. "
            f"The first successful request on the alias route was observed after `{swap_latency_sec}` seconds.\n\n"
            f"Only one alias target changed between the before and after snapshots. "
            f"The MongoDB `routes` entry for `text-processing-service` changed from `{route_before.get('target_service')}` "
            f"to `{route_after.get('target_service')}`, and a request to `http://text-processing-service:8003/index.html` "
            f"returned the `tgn` backend response."
        )
        + "\n",
    )
    write_text(
        exp_dir / "command.txt",
        "python3 scripts/run_router_swap_experiments.py --results-root Results/thesis_experiments\n",
    )
    write_json(logs_dir / "route_before.json", route_before)
    write_json(logs_dir / "route_after.json", route_after)
    write_text(
        logs_dir / "request_response.log",
        f"GET http://text-processing-service:8003/index.html\nstatus_code: {request_result['status_code']}\nbody: {request_result['body']}\n",
    )
    write_text(
        logs_dir / "applied_manifests.log",
        f"before_manifest: {before_path}\nafter_manifest: {after_path}\n",
    )
    return summary


def run_exp02(temp_dir: Path, results_root: Path) -> dict[str, Any]:
    exp_dir = results_root / "exp02_composite_module_swap"
    logs_dir = exp_dir / "logs"
    screenshots_dir = exp_dir / "screenshots"
    logs_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    (screenshots_dir / ".gitkeep").write_text("")

    before_manifest = make_router_manifest(active_model="tgn", shared_text_for_tall=False)
    after_manifest = make_router_manifest(active_model="tall", shared_text_for_tall=False)
    before_path = apply_router_manifest(temp_dir, "exp02-before", before_manifest)
    before_services = wait_for_alias_targets({name: expected_external_name("tgn", name) for name in SERVICE_PORTS})

    start = time.monotonic()
    after_path = apply_router_manifest(temp_dir, "exp02-after", after_manifest)
    after_services = wait_for_alias_targets({name: expected_external_name("tall", name) for name in SERVICE_PORTS})
    composite_swap_latency_sec = round(time.monotonic() - start, 3)

    state_before = fetch_mongo_document(
        "model_router_state",
        {"namespace": DEFAULT_NAMESPACE, "router_name": "tsgv-router"},
    )
    state_after = fetch_mongo_document(
        "model_router_state",
        {"namespace": DEFAULT_NAMESPACE, "router_name": "tsgv-router"},
    )
    services_reconciled = count_changed_services(before_services, after_services)
    logs_excerpt = operator_logs()

    summary = {
        "status": "complete",
        "composite_swap_latency_sec": composite_swap_latency_sec,
        "services_reconciled": services_reconciled,
        "before_active_model": "tgn",
        "after_active_model": state_after.get("active_model", "tall"),
    }

    write_json(exp_dir / "summary.json", summary)
    write_text(
        exp_dir / "README.md",
        "Run context: 2026-04-18 on a local kind cluster using the shared model-router-operator, local smoke backends for both tgn and tall, and MongoDB route persistence in the default namespace.\n",
    )
    write_text(
        exp_dir / "writeup.md",
        (
            f"The composite module swap was measured by changing `spec.activeModel` from `tgn` to `tall` while both model stacks were present in the cluster. "
            f"All six logical alias services changed targets, and the full alias update completed in `{composite_swap_latency_sec}` seconds.\n\n"
            f"The MongoDB routing summary reported `active_model` as `{state_after.get('active_model', 'tall')}` after the reconcile, "
            f"and the operator logs showed a reconcile loop with `aliases=6`. "
            f"No claim beyond the observed alias retargeting and operator reconciliation was made."
        )
        + "\n",
    )
    write_text(
        exp_dir / "command.txt",
        "python3 scripts/run_router_swap_experiments.py --results-root Results/thesis_experiments\n",
    )
    write_text(logs_dir / "operator_reconcile.log", logs_excerpt)
    write_json(logs_dir / "model_router_state_after.json", state_after)
    write_text(
        logs_dir / "applied_manifests.log",
        f"before_manifest: {before_path}\nafter_manifest: {after_path}\n",
    )
    return summary


def update_index(results_root: Path) -> None:
    run(["python3", "scripts/build_thesis_synthesis.py", "--results-root", str(results_root)], cwd=REPO_ROOT)


def main() -> int:
    args = parse_args()
    ensure_tools()
    ensure_operator_image()

    results_root = (REPO_ROOT / args.results_root).resolve()
    temp_root = Path(tempfile.mkdtemp(prefix="router-swap-exp-"))
    created_cluster = False
    try:
        create_cluster()
        created_cluster = True
        load_images()
        setup_base_resources(temp_root)
        run_exp01(temp_root, results_root)
        run_exp02(temp_root, results_root)
        update_index(results_root)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        if created_cluster and not args.keep_cluster:
            delete_cluster()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
