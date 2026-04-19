#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def discover_experiments(results_root: Path) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for exp_dir in sorted(path for path in results_root.iterdir() if path.is_dir() and path.name.startswith("exp")):
        summary_path = exp_dir / "summary.json"
        writeup_path = exp_dir / "writeup.md"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        experiments.append(
            {
                "name": exp_dir.name,
                "dir": exp_dir,
                "summary": summary,
                "writeup_path": writeup_path,
            }
        )
    return experiments


def status_for(summary: dict[str, Any]) -> str:
    raw = str(summary.get("status", "partial")).strip().lower()
    if raw in {"complete", "partial", "deferred"}:
        return raw
    return "partial"


def headline_number_for(summary: dict[str, Any]) -> str:
    if "smoke_checks_passed" in summary:
        return f"smoke_checks_passed: {summary['smoke_checks_passed']}"

    for key in ["max_outbound_peer_count", "max_inbound_peer_count"]:
        value = summary.get(key)
        if value not in {None, "not_run"}:
            return f"{key}: {value}"

    for key in ["swap_latency_sec", "composite_swap_latency_sec", "wallclock_sec", "time_minutes"]:
        value = summary.get(key)
        if value not in {None, "not_run"}:
            return f"{key}: {value}"

    services = summary.get("services")
    if isinstance(services, list) and services:
        endpoint_counts = [int(item.get("endpoints", 0)) for item in services if isinstance(item, dict)]
        if endpoint_counts:
            return f"max_endpoints: {max(endpoint_counts)}"

    changes = summary.get("changes")
    if isinstance(changes, list) and changes:
        files_touched = [int(item.get("files_touched", 0)) for item in changes if isinstance(item, dict)]
        if files_touched:
            return f"max_files_touched: {max(files_touched)}"

    artifacts = summary.get("artifacts")
    if isinstance(artifacts, list):
        return f"artifacts_inventoried: {len(artifacts)}"

    return "not_applicable"


def one_line_outcome_for(exp_name: str, summary: dict[str, Any]) -> str:
    status = status_for(summary)
    reason = str(summary.get("reason", "")).strip()

    if exp_name == "exp03_cross_model_benchmark" and "smoke_checks_passed" in summary:
        return (
            "Full TGN workflow benchmarking was deferred, but "
            f"{summary['smoke_checks_passed']} local TGN smoke checks passed."
        )
    if exp_name == "exp01_single_component_swap" and status == "complete":
        return "One logical alias was retargeted and the swapped route responded successfully."
    if exp_name == "exp02_composite_module_swap" and status == "complete":
        return "All six logical alias services were retargeted during the activeModel swap."
    if exp_name == "exp04_cohesion_analysis":
        return "Endpoint counts and service responsibility inventory were generated."
    if exp_name == "exp05_replacement_effort" and summary.get("status") == "complete":
        return "The visual encoder seam was extended with a replacement-ready resnet18 implementation and smoke-validated."
    if exp_name == "exp06_change_impact":
        changes = summary.get("changes", [])
        return f"{len(changes)} historical TGN-relevant changes were quantified from scoped Git shortstats."
    if exp_name == "exp07_reuse_inventory":
        artifacts = summary.get("artifacts", [])
        if "runtime_reuse_checks" in summary:
            return f"{len(artifacts)} shared artifacts were inventoried and runtime reuse checks were measured."
        return f"{len(artifacts)} shared TGN and routing artifacts were inventoried."
    if exp_name == "exp09_coupling_dependency_assessment":
        return "Static TGN service coupling and dependency edges were enumerated."
    if exp_name in {"exp10_kubernetes_autoscaling_behavior", "exp11_horizontal_scaling_concurrent_requests"}:
        reason = str(summary.get("reason", "")).strip()
        if reason:
            first_sentence = reason.split(".")[0].strip()
            return first_sentence + ("." if not first_sentence.endswith(".") else "")
        return "Kubernetes scaling measurements were deferred."

    if reason:
        first_sentence = reason.split(".")[0].strip()
        return first_sentence + ("." if not first_sentence.endswith(".") else "")

    if status == "complete":
        return "Completed."
    if status == "deferred":
        return "Deferred."
    return "Partially completed."


def render_index(experiments: list[dict[str, Any]]) -> str:
    lines = [
        "| experiment | status | one_line_outcome | headline_number |",
        "| --- | --- | --- | --- |",
    ]
    for exp in experiments:
        summary = exp["summary"]
        lines.append(
            f"| {exp['name']} | {status_for(summary)} | {one_line_outcome_for(exp['name'], summary)} | {headline_number_for(summary)} |"
        )
    return "\n".join(lines) + "\n"


def load_rq_config(config_path: Path | None) -> list[dict[str, Any]]:
    if config_path is None or not config_path.exists():
        return [
            {"id": f"RQ{index}", "text": "RQ text not configured.", "evidence": [], "assessment": "does_not_address"}
            for index in range(1, 6)
        ]

    payload = load_json(config_path)
    sections = payload.get("rq_sections", [])
    if not isinstance(sections, list):
        raise ValueError("rq config must contain a list under 'rq_sections'")
    return [section for section in sections if isinstance(section, dict)]


def resolve_rq_config_path(results_root: Path, explicit_path: str | None) -> Path | None:
    if explicit_path:
        return Path(explicit_path)

    preferred = results_root / "rq_config.json"
    if preferred.exists():
        return preferred

    example = results_root / "rq_config.example.json"
    if example.exists():
        return example

    return None


def render_rq_synthesis(sections: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for section in sections:
        rq_id = str(section.get("id", "RQ")).strip()
        rq_text = str(section.get("text", "RQ text not configured.")).strip()
        evidence = section.get("evidence", [])
        assessment = str(section.get("assessment", "does_not_address")).strip()

        evidence_text = ", ".join(f"`{item}`" for item in evidence) if isinstance(evidence, list) and evidence else "[none]"
        blocks.extend(
            [
                f"**{rq_id}**",
                rq_text,
                f"Evidence files: {evidence_text}.",
                f"Assessment: `{assessment}`",
                "",
            ]
        )
    return "\n".join(blocks).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate thesis_experiments INDEX.md and rq_synthesis.md.")
    parser.add_argument(
        "--results-root",
        default="Results/thesis_experiments",
        help="Path to the thesis experiments bundle root.",
    )
    parser.add_argument(
        "--rq-config",
        default=None,
        help="Optional JSON config containing real RQ text and evidence mappings.",
    )
    parser.add_argument("--skip-index", action="store_true", help="Do not rewrite INDEX.md.")
    parser.add_argument("--skip-rq", action="store_true", help="Do not rewrite rq_synthesis.md.")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    experiments = discover_experiments(results_root)
    if not experiments:
        raise RuntimeError(f"no experiment folders with summary.json found under {results_root}")

    if not args.skip_index:
        index_path = results_root / "INDEX.md"
        index_path.write_text(render_index(experiments))
        print(f"wrote {index_path}")

    if not args.skip_rq:
        rq_path = results_root / "rq_synthesis.md"
        sections = load_rq_config(resolve_rq_config_path(results_root, args.rq_config))
        rq_path.write_text(render_rq_synthesis(sections))
        print(f"wrote {rq_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
