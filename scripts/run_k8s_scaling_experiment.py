#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def command_available(name: str) -> bool:
    return shutil.which(name) is not None


def run_command(command: list[str]) -> tuple[int, str, str]:
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or defer a Kubernetes scaling experiment.")
    parser.add_argument("--experiment", required=True, help="Experiment folder name.")
    parser.add_argument("--service", required=True, help="Logical service under test.")
    parser.add_argument("--output", required=True, help="Path to write the summary JSON.")
    parser.add_argument("--namespace", default="default", help="Kubernetes namespace.")
    args = parser.parse_args()

    summary = {
        "status": "deferred",
        "service": args.service,
        "reason": "",
        "baseline_single_replica": "not_run",
        "hpa_enabled": "not_run",
        "stage_results": [],
    }

    if not command_available("kubectl"):
        summary["reason"] = "kubectl was not installed on this host, so Kubernetes autoscaling measurements could not be executed."
    else:
        code, _, stderr = run_command(["kubectl", "top", "pods", "-n", args.namespace])
        if code != 0:
            detail = stderr.strip() or "kubectl top pods failed"
            summary["reason"] = f"kubectl was available, but metrics-server or pod metrics were not ready: {detail}"
        else:
            summary["reason"] = "Scaling deployment manifests and cluster service endpoints were not configured on this host, so the scripted experiment was not executed."

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
