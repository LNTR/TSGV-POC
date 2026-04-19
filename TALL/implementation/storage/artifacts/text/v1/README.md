# Text Artifact Folder

Place shared text files for `text-processing-service`, `training-service`, and `inference-service` here:

- `metadata.json`
- `sentence_lookup.json`
- `sentence_embeddings.npy`

`TALL` now expects original-style precomputed sentence embeddings instead of token embeddings.

Recommended `metadata.json` shape:

```json
{
  "representation_type": "skip_thought_lookup",
  "embedding_dim": 4800,
  "lookup_file": "sentence_lookup.json",
  "embeddings_file": "sentence_embeddings.npy"
}
```

`sentence_lookup.json` should map lookup keys to row indices in `sentence_embeddings.npy`.

Useful lookup keys are:

- the normalized query text
- `video_name#row:<row_index>`
- `video_name-e<row_number>`

Legacy `vocab.json` + `embeddings.npy` token artifacts are no longer the primary TALL path.
