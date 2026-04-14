# Implementation

This folder contains the standalone microservice-style proof of concept for the TSGV/TGN pipeline. The services share files through `implementation/storage/` and communicate over HTTP with FastAPI.

MongoDB is now included alongside Redis as a service registry index:

- each runtime service self-registers its logical role, model name, implementation name, and compatibility metadata into MongoDB on startup
- the current stack registers the TGN implementations by default
- future model-specific stacks such as `foobar-*` can reuse the same registration pattern and mark shared services with compatible model lists
- each service exposes `GET /registry/self` so you can inspect the exact registration payload it published

## Services

- `visual-preprocessing-service`
  - Input: raw video
  - Output: `shared://frames/processed/<base_name>.vp.npy`
  - Endpoint: `POST /jobs/preprocess`
  - Exact visual reuse:
    - hashes raw video bytes and preprocessing settings
    - reuses matching `vp.npy` artifacts across duplicate videos, even under different file names
    - writes metadata under `shared://frames/metadata/<base_name>.vp.meta.json`

- `visual-feature-extraction-service`
  - Input: `*.vp.npy` frames artifact
  - Output: `shared://features/visual/<base_name>.vf.pt`
  - Endpoint: `POST /jobs/features`
  - Exact feature reuse:
    - reuses matching `vf.pt` artifacts using preprocessing hash + encoder + normalization settings
    - writes metadata under `shared://features/metadata/<base_name>.vf.meta.json`

- `text-processing-service`
  - Input: raw text file or `.aligned.tsv` file plus shared text artifact
  - Output: `shared://text/processed/<base_name>.tp.json`
  - Endpoints: `POST /jobs/process-text`, `POST /jobs/process-aligned-text`, `POST /text/batch`
  - Prediction-only Redis cache:
    - exact normalized text reuse for timestamp-free prediction requests
    - semantic-neighbor lookup for similar prediction requests
    - implemented as composable prediction decorators around the base text processor
    - training TSV processing remains unchanged

- `training-service`
  - Input: split files or a single `vf` + `tp` pair
  - Output: trained model, optimizer state, metrics JSON
  - Endpoints: `POST /jobs/train`, `POST /jobs/train-from-artifacts`

- `inference-service`
  - Input: trained model, `vf` artifact, `tp` artifact
  - Output: predicted temporal segments
  - Endpoint: `POST /infer/ground`

- `evaluation-service`
  - Evaluates held-out prediction records or inference-ready manifests
  - Endpoint: `POST /jobs/evaluate`

## Quick Start

Create one Python environment for all services:

```bash
cd implementation
python -m venv .venv
source .venv/bin/activate
for f in */requirements.txt; do pip install -r "$f"; done
```

By default every service reads and writes under `implementation/storage`. You can keep that default, or set it explicitly:

```bash
export IMPLEMENTATION_STORAGE_ROOT="$PWD/storage"
```

Generated runtime outputs under `storage/` such as frames, visual features, split manifests, processed text, models, and inference/evaluation results are treated as local artifacts and are ignored by this implementation's `.gitignore`.

## Experiment Helpers

The previous ad hoc experiment runner scripts and their sample outputs have been removed. Use the individual services and split manifests directly for training, inference, and evaluation workflows.

For small smoke runs, prefer creating temporary manifests under `storage/splits/` and deleting the generated `storage/frames/`, `storage/features/`, `storage/text/processed/`, `storage/models/`, and `storage/results/` artifacts afterward.

## Run The Services

If you use Docker Compose, the stack also includes a Redis container for prediction-side text caching.

It now also includes a MongoDB container for the service registry index.

Run each service from its own directory. These ports are only suggested local defaults.

```bash
cd implementation/visual-preprocessing-service
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

```bash
cd implementation/visual-feature-extraction-service
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

```bash
cd implementation/text-processing-service
uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
```

```bash
cd implementation/training-service
uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload
```

```bash
cd implementation/inference-service
uvicorn app.main:app --host 0.0.0.0 --port 8005 --reload
```

```bash
cd implementation/evaluation-service
uvicorn app.main:app --host 0.0.0.0 --port 8006 --reload
```

When `MONGODB_URI` is set, each service upserts a document in the `service_registry.services` collection. The default Compose stack sets:

- `SERVICE_MODEL_NAME=tgn`
- `SERVICE_IMPLEMENTATION_NAME=tgn-<service-name>`
- `SERVICE_SELF_URL=http://<compose-service>:<port>`

For Kubernetes-based active model switching, see [`kubernetes/README.md`](../kubernetes/README.md).

For local GPU-backed runs, start Compose with the override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

That override enables CUDA for visual feature extraction, training, and inference.

## Where To Put Inputs

The simplest approach is to use `shared://` URIs. These resolve relative to `implementation/storage/`.

- Raw videos:
  - Keep under `storage/videos/raw/`
  - Example URI: `shared://videos/raw/demo.mp4`

- Raw text files:
  - Keep under `storage/text/raw/`
  - Example URI: `shared://text/raw/demo.txt`

- Shared text artifact:
  - Keep under `storage/artifacts/text/v1/`
  - Required files:
    - `vocab.json`
    - `embeddings.npy`
  - Example URI: `shared://artifacts/text/v1`

- Split files for training/evaluation:
  - Keep under `storage/splits/train/`, `storage/splits/val/`, `storage/splits/test/`
  - Example URI: `shared://splits/train/sample5_train.json`

- Trained models:
  - Usually written under `storage/models/`
  - Example URI: `shared://models/tgn.bin`

You can also pass absolute paths or `file://` URIs instead of `shared://`.

The repo-root `dataset/` folder is optional sample data for local experimentation. It is not used automatically by `shared://` URIs unless you copy files into `storage/` or reference `dataset/...` with an absolute path or `file://`.

See [storage/README.md](storage/README.md) for the storage layout.

## What To Keep And What You Can Remove

Keep these files and folders:

- All service source code folders
- `dataset/videos/*` and `dataset/texts/*` at the repo root if you want to keep the bundled sample data
- `storage/README.md`
- All `.gitkeep` files under `storage/`
- `storage/artifacts/text/v1/vocab.json`
- `storage/artifacts/text/v1/embeddings.npy`
- Raw inputs you still want to reuse:
  - `storage/videos/raw/*`
  - `storage/text/raw/*`
- Split files you still want to reuse:
  - `storage/splits/train/*.json`
  - `storage/splits/val/*.json`
  - `storage/splits/test/*.json`
- Models you want to keep using:
  - `storage/models/*.bin`
  - matching `*.optim`
  - matching `*.metrics.json`

Usually safe to remove when you want to clean generated outputs:

- `storage/frames/processed/*.vp.npy`
- `storage/features/visual/*.vf.pt`
- `storage/text/processed/*.tp.json`
- `storage/results/dispatch/*.json`
- `storage/results/inference/*`
- `storage/results/evaluation/*`

Only remove raw videos, raw text files, split files, or models if you no longer need them. Do not delete `.gitkeep` or the README files if you want to preserve the folder structure in Git.

## How To Give Inputs

All services accept JSON request bodies.

### 1. Preprocess A Video

This reads a raw video, writes `*.vp.npy`, and forwards it to the feature service.

```bash
curl -X POST http://127.0.0.1:8001/jobs/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "input_video_uri": "shared://videos/raw/demo.mp4",
    "output_frame_size": 224,
    "sample_every_sec": 5,
    "feature_service_url": "http://127.0.0.1:8002"
  }'
```

Optional chaining fields:

- `downstream_mode`: `"training"` or `"inference"`
- `downstream_service_url`: training or inference service base URL
- `model_uri`: required when `downstream_mode` is `"inference"`
- `output_model_uri`: required when `downstream_mode` is `"training"`
- `top_n`: inference top-k segments to return
- `hyperparams`: training hyperparameter object

### 2. Process A Text File

This tokenizes the text file and writes `*.tp.json`.

```bash
curl -X POST http://127.0.0.1:8003/jobs/process-text \
  -H "Content-Type: application/json" \
  -d '{
    "input_text_uri": "shared://text/raw/demo.txt",
    "artifact_uri": "shared://artifacts/text/v1",
    "start_time": 12.5,
    "end_time": 19.0
  }'
```

Notes:

- For training, `start_time` and `end_time` must be numeric.
- For inference-only text inputs, `start_time` and `end_time` may be `null`.
- When Redis is configured, timestamp-free prediction requests can reuse cached tokenization for exact normalized text matches.
- The generated `tp.json` must contain `artifact_uri`, `token_ids`, and `length_t`.
- If the text belongs to an event cut from a larger video, you can also pass `base_name` and `video_features_uri` explicitly so the text artifact does not need to share the same base name as the video features.

Prediction cache notes:

- Cache hits are exact after normalization, not semantic.
- `"dog is running"` and `" Dog   is running "` can reuse the same cached tokenization.
- `"dog is walking fast"` will still be processed as a new query.
- When semantic lookup is enabled, the response metadata may also include a `semantic_match` object with the closest previously cached normalized query and similarity score.
- Semantic matches are advisory only in this implementation. New text is still tokenized and written to its own `tp.json`.
- The response metadata includes `cache_hit`, `cache_key`, `cache_enabled`, `semantic_enabled`, and `semantic_vector_ready`.

### 2a. Process An Aligned TSV File

Use this when you want to preserve timestamped event annotations from files such as `dataset/texts/s13-d21.aligned.tsv` in the repo root.

This endpoint:

- reads the TSV row by row
- creates one `tp.json` per aligned event
- keeps `start_frame`, `end_frame`, `start_time`, and `end_time`
- can optionally write a split JSON file for training

Example:

```bash
curl -X POST http://127.0.0.1:8003/jobs/process-aligned-text \
  -H "Content-Type: application/json" \
  -d '{
    "input_alignment_uri": "file:///absolute/path/to/dataset/texts/s13-d21.aligned.tsv",
    "artifact_uri": "shared://artifacts/text/v1",
    "video_features_uri": "shared://features/visual/s13-d21.vf.pt",
    "fps": 30,
    "output_split_uri": "shared://splits/train/s13-d21_events.json"
  }'
```

Optional fields:

- `row_indices`: process only selected zero-based TSV rows
- `base_name_prefix`: override the default event prefix
- `output_split_uri`: write a split JSON that points at the generated event-level `tp.json` files

Generated event artifacts include:

- `source_alignment_uri`
- `row_index`
- `query_text`
- `start_frame`
- `end_frame`
- `start_time`
- `end_time`
- `video_features_uri`

### 3. Train From Split Files

Each split file must be a JSON list. Every item should contain:

- `base_name`
- `video_features_uri`
- `text_processed_uri`

Example split entry:

```json
[
  {
    "base_name": "s13-d21",
    "video_features_uri": "shared://features/visual/s13-d21.vf.pt",
    "text_processed_uri": "shared://text/processed/s13-d21.tp.json"
  }
]
```

Training request example:

```bash
curl -X POST http://127.0.0.1:8004/jobs/train \
  -H "Content-Type: application/json" \
  -d '{
    "train_split_uri": "shared://splits/train/sample5_train.json",
    "val_split_uri": "shared://splits/val/sample5_val.json",
    "features_root_uri": "shared://features/visual",
    "output_model_uri": "shared://models/tgn.bin",
    "hyperparams": {
      "K": 16,
      "delta": 2,
      "threshold": 0.5,
      "batch_size": 1,
      "lr": 0.001
    }
  }'
```

Important training requirements:

- Every referenced `text_processed_uri` must exist.
- Every referenced `video_features_uri` must exist.
- Training `tp.json` files must include numeric `start_time` and `end_time`.
- Event-level `tp.json` files generated from aligned TSV data are supported.
- All training records must point to the same shared text artifact directory.

### 4. Train From One Pair Of Artifacts

Useful for quick tests:

```bash
curl -X POST http://127.0.0.1:8004/jobs/train-from-artifacts \
  -H "Content-Type: application/json" \
  -d '{
    "video_features_uri": "shared://features/visual/demo.vf.pt",
    "text_processed_uri": "shared://text/processed/demo.tp.json",
    "output_model_uri": "shared://models/demo.bin",
    "hyperparams": {
      "K": 16,
      "delta": 2,
      "threshold": 0.5,
      "batch_size": 1,
      "lr": 0.001
    }
  }'
```

### 5. Run Inference

```bash
curl -X POST http://127.0.0.1:8005/infer/ground \
  -H "Content-Type: application/json" \
  -d '{
    "model_uri": "shared://models/tgn.bin",
    "video_features_uri": "shared://features/visual/demo.vf.pt",
    "text_processed_uri": "shared://text/processed/demo.tp.json",
    "top_n": 5
  }'
```

The `text_processed_uri` used for inference must point to a `tp.json` file that contains:

- `artifact_uri`
- `token_ids`
- `length_t`

## Recommended Local Workflow

1. Put `vocab.json` and `embeddings.npy` in `storage/artifacts/text/v1/`.
2. Put raw videos in `storage/videos/raw/`.
3. Put raw text files in `storage/text/raw/`.
4. Start the services you need.
5. Call `text-processing-service` to create `*.tp.json`.
6. Call `visual-preprocessing-service` to create `*.vp.npy` and `*.vf.pt`.
7. Call `training-service` or `inference-service` once both text and visual artifacts exist.

Text and visual preparation can run in parallel. They join at training or inference time.

## Notes

- Existing generated artifacts are overwritten when rerun.
- The implementation is meant for local experimentation, not production hardening.
- `evaluation-service` is present but still minimal compared with the other services.
