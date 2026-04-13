#!/usr/bin/env python
import argparse
import json
import math
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path


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


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def split_video_ids(video_ids: list[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    ordered = list(video_ids)
    rng.shuffle(ordered)

    total = len(ordered)
    train_end = math.floor(total * 0.6)
    val_end = train_end + math.floor(total * 0.2)
    train = sorted(ordered[:train_end])
    val = sorted(ordered[train_end:val_end])
    test = sorted(ordered[val_end:])
    return train, val, test


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

        extract_visual_features(
            frames_uri=preprocess["frames_uri"],
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

    return features_uri


def process_aligned_video(
    implementation_root: Path,
    video_id: str,
    video_features_uri: str,
) -> list[dict]:
    alignment_path = implementation_root / "dataset" / "texts" / f"{video_id}.aligned.tsv"
    if not alignment_path.exists():
        raise FileNotFoundError(f"missing aligned text file: {alignment_path}")

    with service_import_path(implementation_root / "text-processing-service"):
        from app.main import ProcessAlignedTextRequest, process_aligned_text

        response = process_aligned_text(
            ProcessAlignedTextRequest(
                input_alignment_uri=absolute_input(alignment_path),
                artifact_uri=shared_uri("artifacts", "text", "v1"),
                video_features_uri=video_features_uri,
                base_name_prefix=video_id,
            )
        )

    return response.records


def build_inference_text_artifact(
    implementation_root: Path,
    storage_root: Path,
    tag: str,
    record: dict,
) -> str:
    raw_text_rel = Path("text") / "raw" / tag / "test" / f"{record['base_name']}.txt"
    raw_text_path = storage_root / raw_text_rel
    raw_text_path.parent.mkdir(parents=True, exist_ok=True)
    raw_text_path.write_text(record["query_text"] + "\n")

    with service_import_path(implementation_root / "text-processing-service"):
        from app.main import ProcessTextFileRequest, process_text_file

        response = process_text_file(
            ProcessTextFileRequest(
                input_text_uri=shared_uri(*raw_text_rel.parts),
                artifact_uri=shared_uri("artifacts", "text", "v1"),
                start_time=None,
                end_time=None,
                base_name=f"{record['base_name']}-infer",
                video_features_uri=record["video_features_uri"],
            )
        )

    return str(response.metadata["text_processed_uri"])


def prepare_dataset(
    implementation_root: Path,
    storage_root: Path,
    tag: str,
    seed: int,
    rebuild_features: bool,
    frame_size: int,
    sample_every_sec: int,
) -> dict:
    video_ids = sorted(p.stem for p in (implementation_root / "dataset" / "videos").glob("*.avi"))
    train_videos, val_videos, test_videos = split_video_ids(video_ids, seed=seed)
    split_by_video = {
        "train": set(train_videos),
        "val": set(val_videos),
        "test": set(test_videos),
    }

    split_records = {"train": [], "val": [], "test": []}
    test_queries = []

    total = len(video_ids)
    for index, video_id in enumerate(video_ids, start=1):
        features_uri = ensure_visual_features(
            implementation_root=implementation_root,
            storage_root=storage_root,
            video_id=video_id,
            rebuild_features=rebuild_features,
            frame_size=frame_size,
            sample_every_sec=sample_every_sec,
        )
        event_records = process_aligned_video(
            implementation_root=implementation_root,
            video_id=video_id,
            video_features_uri=features_uri,
        )

        split_name = "train"
        if video_id in split_by_video["val"]:
            split_name = "val"
        elif video_id in split_by_video["test"]:
            split_name = "test"

        for record in event_records:
            split_records[split_name].append(
                {
                    "base_name": record["base_name"],
                    "video_features_uri": record["video_features_uri"],
                    "text_processed_uri": record["text_processed_uri"],
                }
            )
            if split_name == "test":
                test_queries.append(
                    {
                        "base_name": record["base_name"],
                        "video_id": video_id,
                        "query_text": record["query_text"],
                        "gold_start": record["start_time"],
                        "gold_end": record["end_time"],
                        "video_features_uri": record["video_features_uri"],
                        "train_text_processed_uri": record["text_processed_uri"],
                    }
                )

        if index % 10 == 0 or index == total:
            print(
                f"[prepare] videos {index}/{total} "
                f"train={len(split_records['train'])} "
                f"val={len(split_records['val'])} "
                f"test={len(split_records['test'])}"
            , flush=True)

    for row in test_queries:
        row["inference_text_processed_uri"] = build_inference_text_artifact(
            implementation_root=implementation_root,
            storage_root=storage_root,
            tag=tag,
            record=row,
        )

    return {
        "split_videos": {
            "train": train_videos,
            "val": val_videos,
            "test": test_videos,
        },
        "split_records": split_records,
        "test_queries": test_queries,
    }


def run_training(
    implementation_root: Path,
    train_split_uri: str,
    val_split_uri: str,
    model_uri: str,
    max_iter: int,
    batch_size: int,
    sample_rate: int,
) -> dict:
    with service_import_path(implementation_root / "training-service"):
        from app.trainer import HyperParams, run_training_job

        hyperparams = HyperParams(
            k_scales=16,
            delta=2,
            threshold=0.5,
            batch_size=batch_size,
            lr=0.001,
            log_every=5,
            max_iter=max_iter,
            valid_niter=20,
            top_n_eval=1,
            patience=2,
            max_num_trial=3,
            fps=30,
            sample_rate=sample_rate,
        )
        return run_training_job(
            train_split_uri=train_split_uri,
            val_split_uri=val_split_uri,
            features_root_uri=shared_uri("features", "visual"),
            output_model_uri=model_uri,
            hyperparams=hyperparams,
        )


def run_test_inference(
    implementation_root: Path,
    model_uri: str,
    test_queries: list[dict],
    top_n: int,
    sample_rate: int,
    fps: int,
) -> list[dict]:
    with service_import_path(implementation_root / "inference-service"):
        from app.pipeline import infer_segments

        evaluation_records = []
        for index, row in enumerate(test_queries, start=1):
            pred_segments = infer_segments(
                model_uri=model_uri,
                video_features_uri=row["video_features_uri"],
                text_processed_uri=row["inference_text_processed_uri"],
                top_n=top_n,
                sample_rate=sample_rate,
                fps=fps,
            )
            evaluation_records.append(
                {
                    "base_name": row["base_name"],
                    "video_id": row["video_id"],
                    "query_text": row["query_text"],
                    "gold_start": row["gold_start"],
                    "gold_end": row["gold_end"],
                    "pred_segments": [
                        {
                            "start": seg["start_sec"],
                            "end": seg["end_sec"],
                            "score": seg["score"],
                        }
                        for seg in pred_segments
                    ],
                }
            )
            if index % 100 == 0 or index == len(test_queries):
                print(f"[infer] test queries {index}/{len(test_queries)}", flush=True)
        return evaluation_records


def run_evaluation(
    implementation_root: Path,
    model_uri: str,
    test_predictions_uri: str,
    metrics: list[str],
) -> dict[str, float]:
    with service_import_path(implementation_root / "evaluation-service"):
        from app.evaluator import run_evaluation_job

        return run_evaluation_job(
            model_uri=model_uri,
            test_split_uri=test_predictions_uri,
            metrics=metrics,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 60/20/20 dataset split experiment on implementation/dataset."
    )
    parser.add_argument("--tag", default="dataset_split_60_20_20", help="Output name prefix under implementation/storage.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the video-level split.")
    parser.add_argument("--max-iter", type=int, default=60, help="Training iterations.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--top-n", type=int, default=5, help="Predicted segments per test query.")
    parser.add_argument("--frame-size", type=int, default=224, help="Video frame size for preprocessing.")
    parser.add_argument("--sample-every-sec", type=int, default=5, help="Seconds between sampled frames.")
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Regenerate visual features from dataset videos even if shared artifacts already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    implementation_root = Path(__file__).resolve().parent
    storage_root = Path(
        os.environ.setdefault("IMPLEMENTATION_STORAGE_ROOT", str(implementation_root / "storage"))
    ).resolve()
    fps = 30
    sample_rate = fps * args.sample_every_sec

    dataset_info = prepare_dataset(
        implementation_root=implementation_root,
        storage_root=storage_root,
        tag=args.tag,
        seed=args.seed,
        rebuild_features=args.rebuild_features,
        frame_size=args.frame_size,
        sample_every_sec=args.sample_every_sec,
    )

    split_paths = {
        "train": storage_root / "splits" / "train" / f"{args.tag}_train.json",
        "val": storage_root / "splits" / "val" / f"{args.tag}_val.json",
        "test_predictions": storage_root / "splits" / "test" / f"{args.tag}_test_predictions.json",
    }
    summary_path = storage_root / "results" / f"{args.tag}_summary.json"
    manifest_path = storage_root / "results" / f"{args.tag}_manifest.json"
    model_uri = shared_uri("models", f"{args.tag}.bin")

    write_json(split_paths["train"], dataset_info["split_records"]["train"])
    write_json(split_paths["val"], dataset_info["split_records"]["val"])
    write_json(manifest_path, dataset_info)

    training = run_training(
        implementation_root=implementation_root,
        train_split_uri=shared_uri("splits", "train", split_paths["train"].name),
        val_split_uri=shared_uri("splits", "val", split_paths["val"].name),
        model_uri=model_uri,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        sample_rate=sample_rate,
    )
    evaluation_records = run_test_inference(
        implementation_root=implementation_root,
        model_uri=model_uri,
        test_queries=dataset_info["test_queries"],
        top_n=args.top_n,
        sample_rate=sample_rate,
        fps=fps,
    )
    write_json(split_paths["test_predictions"], evaluation_records)

    metrics = run_evaluation(
        implementation_root=implementation_root,
        model_uri=model_uri,
        test_predictions_uri=shared_uri("splits", "test", split_paths["test_predictions"].name),
        metrics=["R@1_IOU0.5", "R@5_IOU0.5"],
    )

    summary = {
        "tag": args.tag,
        "seed": args.seed,
        "fps": fps,
        "sample_every_sec": args.sample_every_sec,
        "sample_rate": sample_rate,
        "split_video_counts": {key: len(value) for key, value in dataset_info["split_videos"].items()},
        "split_record_counts": {key: len(value) for key, value in dataset_info["split_records"].items()},
        "training": training,
        "test_metrics": metrics,
        "model_uri": model_uri,
        "train_split_uri": shared_uri("splits", "train", split_paths["train"].name),
        "val_split_uri": shared_uri("splits", "val", split_paths["val"].name),
        "test_predictions_uri": shared_uri("splits", "test", split_paths["test_predictions"].name),
    }
    write_json(summary_path, summary)

    print(f"[done] summary: {summary_path}", flush=True)
    print(f"[done] model: {storage_root / 'models' / f'{args.tag}.bin'}", flush=True)
    print(f"[done] metrics: {metrics}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
