# TSGV Model Services

This repository contains two model implementations with the same service layout:

- `TGN/implementation`
- `TALL/implementation`

Each implementation exposes the same six service boundaries:

- `visual-preprocessing-service`
- `visual-feature-extraction-service`
- `text-processing-service`
- `training-service`
- `inference-service`
- `evaluation-service`

Shared API contracts live in [`contracts/`](./contracts).

## Prerequisites

- Docker Engine
- Docker Compose v2
- At least 10 GB free disk space for images, caches, models, and dataset artifacts

Optional for GPU runs:

- NVIDIA GPU
- NVIDIA Container Toolkit installed on the host

## Repo Layout

- [`TGN/implementation`](./TGN/implementation): TGN service stack
- [`TALL/implementation`](./TALL/implementation): TALL service stack
- [`contracts`](./contracts): shared OpenAPI contracts
- [`kubernetes`](./kubernetes): shared model-router operator and manifests

## Important Data Notes

Both stacks expect the dataset and runtime storage to live inside each implementation folder:

- `dataset/`
- `storage/`

The Compose files mount them into the containers as:

- `/app/dataset`
- `/app/storage`

### TGN artifacts

`TGN` expects its text artifact under:

- `TGN/implementation/storage/artifacts/text/v1/`

That folder should contain:

- `vocab.json`
- `embeddings.npy`

### TALL artifacts

`TALL` expects:

- C3D weights at `TALL/implementation/storage/artifacts/video/c3d/c3d.pickle`
- a sentence-embedding text artifact directory referenced by requests at runtime

For the current `TALL` implementation, the text artifact must contain:

- `metadata.json`
- `sentence_lookup.json`
- `sentence_embeddings.npy`

The shipped `TALL/storage/artifacts/text/v1/` folder is not a ready-made production artifact by itself. Replace or populate it with a real sentence-embedding artifact before running real `TALL` text, training, or inference workflows.

## Docker Quick Start

You can run either model stack independently.

### TGN

```bash
cd "TGN/implementation"
docker compose up -d --build
```

Stop it with:

```bash
docker compose down
```

### TALL

```bash
cd "TALL/implementation"
docker compose up -d --build
```

Stop it with:

```bash
docker compose down
```

## Optional GPU Mode

The base Compose files are CPU-safe by default.

If your host has NVIDIA Docker support and you want GPU access, use the GPU override file.

### TGN with GPU feature extraction

```bash
cd "TGN/implementation"
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

`TGN` training remains CPU-oriented in the current Docker setup.

### TALL with GPU feature extraction and GPU training

```bash
cd "TALL/implementation"
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

## Service Ports

Each stack exposes the same host ports:

- `8001`: visual preprocessing
- `8002`: visual feature extraction
- `8003`: text processing
- `8004`: training
- `8005`: inference
- `8006`: evaluation
- `27017`: MongoDB
- `6379`: Redis

Do not run both stacks on the same host at the same time unless you first change one set of exposed ports.

## Health Checks

After startup, you can verify services with:

```bash
curl http://127.0.0.1:8001/registry/self
curl http://127.0.0.1:8002/registry/self
curl http://127.0.0.1:8003/registry/self
curl http://127.0.0.1:8004/registry/self
curl http://127.0.0.1:8005/registry/self
curl http://127.0.0.1:8006/registry/self
```

## Logs

View all logs for the active stack:

```bash
docker compose logs -f
```

View one service:

```bash
docker compose logs -f training-service
```

## First Run Advice

- `TGN` visual feature extraction may download VGG16 weights on first use.
- `TALL` needs valid C3D and sentence-embedding artifacts before real model workflows will succeed.
- `TALL` training can use a lot of memory on small GPUs, so smoke-scale hyperparameters are safer on low-VRAM machines.

## Shared Contracts

The shared service contracts are in:

- [`contracts/visual-preprocessing.openapi.yaml`](./contracts/visual-preprocessing.openapi.yaml)
- [`contracts/visual-feature-extraction.openapi.yaml`](./contracts/visual-feature-extraction.openapi.yaml)
- [`contracts/text-processing.openapi.yaml`](./contracts/text-processing.openapi.yaml)
- [`contracts/training.openapi.yaml`](./contracts/training.openapi.yaml)
- [`contracts/inference.openapi.yaml`](./contracts/inference.openapi.yaml)
- [`contracts/evaluation.openapi.yaml`](./contracts/evaluation.openapi.yaml)
