# Smoke Dataset

This folder contains the minimal dataset subset used by `TGN-smoke/run_compose_smoke_test.py` by default.

Included samples:

- `videos/s37-d25.avi`
- `videos/s22-d55.avi`
- `videos/s21-d50.avi`
- `texts/s37-d25.aligned.tsv`
- `texts/s22-d55.aligned.tsv`
- `texts/s21-d50.aligned.tsv`

These match the script's default smoke videos:

- `s37-d25`
- `s22-d55`
- `s21-d50`

Folder layout matches the main repo `dataset/` structure so it can be mounted as `/app/dataset` if you want to run the smoke test against this smaller subset.
