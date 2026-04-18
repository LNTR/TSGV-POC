# TGN Smoke

This folder contains the host-side smoke-test helper for the `TGN` Docker Compose stack and the per-run files it generates.

## Files

- `run_compose_smoke_test.py`: runs a small `3`-video training, inference, and evaluation smoke flow against the Compose ports.
- `run_compose_dataset_workflow.py`: builds a deterministic `60/20/20` split from the dataset and runs the full Compose workflow on it.
- `<tag>/`: created on demand for each smoke run unless `--cleanup` is used.

## Run

The easiest one-command path is:

```bash
python TGN-smoke/run_compose_smoke_test.py --compose-managed --cleanup
```

That command:

- starts the GPU-enabled `TGN` Compose services
- waits for them to become ready
- runs the smoke test
- removes the generated smoke artifacts
- shuts the Compose services back down

When `--compose-managed` is used, the script now starts the stack with `docker compose ... up --build` by default. That means a fresh machine will build the real service images instead of relying on any prebuilt local tags.

Docker will still reuse its existing build cache layers automatically whenever possible, so repeated runs do not need to redownload or rebuild unchanged layers. The script does not force `--no-cache`.

If you prefer to start Compose manually first, run this from `TGN/implementation/`:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build \
  mongodb redis visual-preprocessing-service visual-feature-extraction-service \
  text-processing-service training-service inference-service evaluation-service
```

Then run the smoke test from the repo root or from this folder:

```bash
python TGN-smoke/run_compose_smoke_test.py --cleanup
```

If you want the script to start Compose but leave the containers up afterward:

```bash
python TGN-smoke/run_compose_smoke_test.py --compose-managed --keep-compose-up
```

If you intentionally want to skip rebuilding and only use whatever images already exist locally:

```bash
python TGN-smoke/run_compose_smoke_test.py --compose-managed --compose-no-build
```

## Full Dataset Workflow

To split the dataset into `60/20/20` and run preprocess, text processing, training, sample inference, and evaluation through the GPU Compose stack:

```bash
python TGN-smoke/run_compose_dataset_workflow.py --compose-managed
```

Useful variants:

```bash
python TGN-smoke/run_compose_dataset_workflow.py --compose-managed --cleanup
python TGN-smoke/run_compose_dataset_workflow.py --compose-managed --compose-no-build
python TGN-smoke/run_compose_dataset_workflow.py --compose-managed --video-limit 30
```

The dataset workflow script:

- scans the repo-root `dataset/` for matching `videos/*.avi` and `texts/*.aligned.tsv`
- shuffles them deterministically with `--seed`
- splits them into `60/20/20` train/val/test video groups
- runs the full Docker Compose pipeline with the GPU override file
- writes host-visible outputs under `TGN-smoke/<tag>/` unless `--cleanup` is used
- records per-phase timings under `phase_timings_sec` in the summary JSON
- records sampled container RAM usage under `phase_memory_stats` in the summary JSON

Memory stats are collected from `docker stats`, so they represent container memory usage, not GPU VRAM. You can change the sampling frequency with `--stats-interval-sec`.

Without `--cleanup`, each run writes its host-managed files under `TGN-smoke/<tag>/`:

- `train.json`
- `val.json`
- `eval.json`
- `query.txt`
- `model.bin`
- `model.bin.optim`
- `model.bin.metrics.json`
- `summary.json`

The script may create a temporary bridge directory under `TGN/implementation/storage/smoke/<tag>/` while the containers are running, but it copies the smoke run files back into `TGN-smoke/<tag>/` and removes that bridge directory before exit.

The running services still write shared artifacts such as `vf.pt` and `tp.json` under `TGN/implementation/storage/`.
