# OpenAPI Contracts

This folder contains shared OpenAPI YAML contracts for the TGN and TALL service boundaries.

Files:
- `visual-preprocessing.openapi.yaml`
- `visual-feature-extraction.openapi.yaml`
- `text-processing.openapi.yaml`
- `training.openapi.yaml`
- `inference.openapi.yaml`
- `evaluation.openapi.yaml`

Notes:
- The contracts now aim for a common service envelope plus typed variants where the models genuinely differ.
- Shared fields stay shared: job metadata, URIs, status, metadata, downstream hints, and artifact references.
- Divergent parts are modeled explicitly:
  - text batch responses use representation variants such as token ids vs sentence embeddings
  - training hyperparameters use model-family variants such as TGN vs TALL
  - visual extraction hints use recipe variants such as frame encoders vs clip encoders
- Legacy flat fields are still present in some request contracts so the current services remain representable while the interfaces move toward the generalized form.
