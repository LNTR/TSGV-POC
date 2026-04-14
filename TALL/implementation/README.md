# TALL Implementation

This folder mirrors the `TGN/implementation` service architecture for `TALL/CTRL`.

It keeps the same service boundaries:

- `visual-preprocessing-service`
- `visual-feature-extraction-service`
- `text-processing-service`
- `training-service`
- `inference-service`
- `evaluation-service`

The shared services remain compatible with the `TGN` layout, while the `training-service` and `inference-service` now use a `TALL`-specific `CTRL`-style temporal localizer.

## Model Behavior

- `text-processing-service` still writes `shared://text/processed/*.tp.json` records using the shared text artifact.
- `TALL` now matches the original paper setup on the text side: each processed query resolves to a precomputed skip-thought-style sentence embedding from `storage/artifacts/text/v1/`.
- `visual-feature-extraction-service` now expects original-style `C3D fc6` weights and produces 4096-d clip features instead of framewise `VGG16` embeddings.
- `training-service` builds sliding temporal candidates with left/right context from each `vf.pt` feature tensor and learns:
  - an alignment score for each candidate
  - start/end boundary offsets for regression
- `inference-service` rebuilds the same candidates, scores them, regresses the boundaries, and returns top-ranked segments after temporal NMS.

## Quick Start

```bash
cd "TALL/implementation"
python -m venv .venv
source .venv/bin/activate
for f in */requirements.txt; do pip install -r "$f"; done
```

Optionally pin the storage root:

```bash
export IMPLEMENTATION_STORAGE_ROOT="$PWD/storage"
```

Generated runtime outputs under `storage/` such as frames, visual features, split manifests, processed text, models, and inference/evaluation results are treated as local artifacts and are ignored by this implementation's `.gitignore`.

## Training Inputs

Training uses the same split manifest shape as `TGN`: a JSON list of records containing:

- `base_name`
- `video_features_uri`
- `text_processed_uri`

Each referenced text-processed record must include:

- `artifact_uri`
- `sentence_embedding`
- `start_time`
- `end_time`

## Service Run Commands

```bash
cd "TALL/implementation/visual-preprocessing-service"
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

```bash
cd "TALL/implementation/visual-feature-extraction-service"
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

```bash
cd "TALL/implementation/text-processing-service"
uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
```

```bash
cd "TALL/implementation/training-service"
uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload
```

```bash
cd "TALL/implementation/inference-service"
uvicorn app.main:app --host 0.0.0.0 --port 8005 --reload
```

```bash
cd "TALL/implementation/evaluation-service"
uvicorn app.main:app --host 0.0.0.0 --port 8006 --reload
```

## Registry Defaults

The local implementation defaults to `SERVICE_MODEL_NAME=tall`.

The included Compose file also registers:

- `SERVICE_IMPLEMENTATION_NAME=tall-<service-name>`
- `SERVICE_SELF_URL=http://<compose-service>:<port>`

## Artifacts

Use `shared://` URIs relative to `storage/`.

- Raw videos: `shared://videos/raw/demo.mp4`
- Raw text: `shared://text/raw/demo.txt`
- Shared text artifact: `shared://artifacts/text/v1`
  - required files: `metadata.json`, `sentence_lookup.json`, `sentence_embeddings.npy`
  - expected representation: precomputed skip-thought-style sentence embeddings, typically `4800`-d for original TALL compatibility
- C3D weights: `shared://artifacts/video/c3d/c3d.pickle`
- Trained model: `shared://models/tall.bin`

The shared sample dataset now lives at the repo root in `dataset/`, and the Compose setup mounts it into containers at `/app/dataset`.
