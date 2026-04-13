# Shared Local Storage

This directory is the default shared storage root used by all services in `implentation/`.

Use `shared://` URIs in API payloads, for example:

- `shared://videos/raw/demo.avi`
- `shared://frames/processed/demo.vp.npy`
- `shared://features/visual/demo.vf.pt`
- `shared://text/processed/demo.tp.json`
- `shared://artifacts/text/v1`
- `shared://models/tgn.bin`
- `shared://results/dispatch/demo.inference.json`

You can override this root with the `IMPLEMENTATION_STORAGE_ROOT` environment variable.
