#!/usr/bin/env python
import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path


SELECTED_VIDEO_IDS = [
    "s13-d21",
    "s13-d25",
    "s13-d28",
    "s13-d31",
    "s13-d40",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def clear_service_modules() -> None:
    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            sys.modules.pop(name, None)


@contextmanager
def service_import_path(service_dir: Path):
    clear_service_modules()
    sys.path.insert(0, str(service_dir))
    try:
        yield
    finally:
        if str(service_dir) in sys.path:
            sys.path.remove(str(service_dir))
        clear_service_modules()


def shared_uri(*parts: str) -> str:
    return "shared://" + "/".join(part.strip("/") for part in parts if part)


def absolute_input(path: Path) -> str:
    return str(path.resolve())


def ensure_dirs(storage_root: Path, tag: str) -> None:
    for rel in [
        "frames/processed",
        "features/visual",
        "models",
        "results",
        "splits/train",
        "splits/val",
        f"text/raw/{tag}",
        "text/processed",
    ]:
        (storage_root / rel).mkdir(parents=True, exist_ok=True)


def ensure_visual_features(
    implementation_root: Path,
    storage_root: Path,
    video_id: str,
    rebuild_features: bool,
    frame_size: int,
    sample_every_sec: int,
) -> str:
    features_uri = shared_uri("features", "visual", f"{video_id}.vf.pt")
    features_path = storage_root / "features" / "visual" / f"{video_id}.vf.pt"
    if features_path.exists() and not rebuild_features:
        print(f"[features] reuse {features_uri}")
        return features_uri

    video_path = implementation_root / "dataset" / "videos" / f"{video_id}.avi"
    if not video_path.exists():
        raise FileNotFoundError(f"missing dataset video: {video_path}")

    with service_import_path(implementation_root / "visual-preprocessing-service"):
        from app.processing import run_preprocess_job

        preprocess = run_preprocess_job(
            input_video_uri=absolute_input(video_path),
            output_frame_size=frame_size,
            sample_every_sec=sample_every_sec,
        )

    with service_import_path(implementation_root / "visual-feature-extraction-service"):
        from app.extraction import extract_visual_features

        feature_metadata = extract_visual_features(
            frames_uri=preprocess["frames_uri"],
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

    print(f"[features] built {feature_metadata['features_uri']} from {video_path.name}")
    return str(feature_metadata["features_uri"])


def build_text_artifacts(
    implementation_root: Path,
    storage_root: Path,
    tag: str,
    video_id: str,
    video_features_uri: str,
) -> dict:
    alignment_path = implementation_root / "dataset" / "texts" / f"{video_id}.aligned.tsv"
    if not alignment_path.exists():
        raise FileNotFoundError(f"missing aligned text file: {alignment_path}")

    with service_import_path(implementation_root / "text-processing-service"):
        from app.main import (
            ProcessAlignedTextRequest,
            ProcessTextFileRequest,
            process_aligned_text,
            process_text_file,
        )

        aligned = process_aligned_text(
            ProcessAlignedTextRequest(
                input_alignment_uri=absolute_input(alignment_path),
                artifact_uri=shared_uri("artifacts", "text", "v1"),
                video_features_uri=video_features_uri,
                row_indices=[0],
                base_name_prefix=video_id,
            )
        )
        record = aligned.records[0]
        raw_text_rel = Path("text") / "raw" / tag / f"{record['base_name']}.txt"
        raw_text_path = storage_root / raw_text_rel
        raw_text_path.write_text(record["query_text"] + "\n")

        inference_base_name = f"{record['base_name']}-infer"
        inference = process_text_file(
            ProcessTextFileRequest(
                input_text_uri=shared_uri(*raw_text_rel.parts),
                artifact_uri=shared_uri("artifacts", "text", "v1"),
                start_time=None,
                end_time=None,
                base_name=inference_base_name,
                video_features_uri=video_features_uri,
            )
        )

    return {
        "base_name": record["base_name"],
        "video_id": video_id,
        "query_text": record["query_text"],
        "train_start_time": record["start_time"],
        "train_end_time": record["end_time"],
        "video_features_uri": video_features_uri,
        "train_text_processed_uri": record["text_processed_uri"],
        "raw_text_uri": shared_uri(*raw_text_rel.parts),
        "inference_base_name": inference_base_name,
        "inference_text_processed_uri": inference.metadata["text_processed_uri"],
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def train_model(
    implementation_root: Path,
    tag: str,
    train_split_uri: str,
    val_split_uri: str,
    output_model_uri: str,
    max_iter: int,
) -> dict:
    with service_import_path(implementation_root / "training-service"):
        from app.trainer import HyperParams, run_training_job

        hyperparams = HyperParams(
            k_scales=16,
            delta=2,
            threshold=0.5,
            batch_size=1,
            lr=0.001,
            log_every=1,
            max_iter=max_iter,
            valid_niter=5,
            top_n_eval=1,
            patience=2,
            max_num_trial=3,
            fps=30,
            sample_rate=150,
        )
        metadata = run_training_job(
            train_split_uri=train_split_uri,
            val_split_uri=val_split_uri,
            features_root_uri=shared_uri("features", "visual"),
            output_model_uri=output_model_uri,
            hyperparams=hyperparams,
        )

    print(
        "[train] completed "
        f"{metadata['iterations_completed']} iterations, model={metadata['model_uri']}"
    )
    return metadata


def run_inference(
    implementation_root: Path,
    model_uri: str,
    manifest: list[dict],
    top_n: int,
) -> list[dict]:
    with service_import_path(implementation_root / "inference-service"):
        from app.pipeline import infer_segments

        results = []
        for row in manifest:
            segments = infer_segments(
                model_uri=model_uri,
                video_features_uri=row["video_features_uri"],
                text_processed_uri=row["inference_text_processed_uri"],
                top_n=top_n,
            )
            results.append(
                {
                    "base_name": row["base_name"],
                    "video_id": row["video_id"],
                    "query_text": row["query_text"],
                    "ground_truth": {
                        "start_sec": row["train_start_time"],
                        "end_sec": row["train_end_time"],
                    },
                    "predicted_segments": segments,
                }
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 5-input training and timestamp-free grounding sanity check."
    )
    parser.add_argument("--tag", default="five_input_sanity", help="Output name prefix under implementation/storage.")
    parser.add_argument("--max-iter", type=int, default=15, help="Training iterations for the sanity run.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of inferred segments to save per query.")
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Regenerate visual features from dataset videos even if shared feature artifacts already exist.",
    )
    parser.add_argument("--frame-size", type=int, default=224, help="Video frame size for preprocessing.")
    parser.add_argument("--sample-every-sec", type=int, default=5, help="Seconds between sampled frames.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    implementation_root = Path(__file__).resolve().parent
    storage_root = Path(
        os.environ.setdefault("IMPLEMENTATION_STORAGE_ROOT", str(implementation_root / "storage"))
    ).resolve()
    ensure_dirs(storage_root, args.tag)

    manifest = []
    for video_id in SELECTED_VIDEO_IDS:
        features_uri = ensure_visual_features(
            implementation_root=implementation_root,
            storage_root=storage_root,
            video_id=video_id,
            rebuild_features=args.rebuild_features,
            frame_size=args.frame_size,
            sample_every_sec=args.sample_every_sec,
        )
        manifest.append(
            build_text_artifacts(
                implementation_root=implementation_root,
                storage_root=storage_root,
                tag=args.tag,
                video_id=video_id,
                video_features_uri=features_uri,
            )
        )

    train_split = [
        {
            "base_name": row["base_name"],
            "video_features_uri": row["video_features_uri"],
            "text_processed_uri": row["train_text_processed_uri"],
        }
        for row in manifest
    ]
    train_split_path = storage_root / "splits" / "train" / f"{args.tag}_train.json"
    val_split_path = storage_root / "splits" / "val" / f"{args.tag}_val.json"
    manifest_path = storage_root / "results" / f"{args.tag}_manifest.json"
    inference_path = storage_root / "results" / f"{args.tag}_inference.json"
    model_uri = shared_uri("models", f"{args.tag}.bin")

    write_json(train_split_path, train_split)
    write_json(val_split_path, [])
    write_json(manifest_path, manifest)

    training = train_model(
        implementation_root=implementation_root,
        tag=args.tag,
        train_split_uri=shared_uri("splits", "train", train_split_path.name),
        val_split_uri=shared_uri("splits", "val", val_split_path.name),
        output_model_uri=model_uri,
        max_iter=args.max_iter,
    )
    inference = run_inference(
        implementation_root=implementation_root,
        model_uri=model_uri,
        manifest=manifest,
        top_n=args.top_n,
    )
    write_json(inference_path, inference)

    print(f"[done] manifest: {manifest_path}")
    print(f"[done] model: {storage_root / 'models' / f'{args.tag}.bin'}")
    print(f"[done] inference: {inference_path}")
    print(f"[done] iterations: {training['iterations_completed']}")
    for row in inference:
        top1 = row["predicted_segments"][0] if row["predicted_segments"] else None
        print(f"[top1] {row['base_name']}: {top1}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
