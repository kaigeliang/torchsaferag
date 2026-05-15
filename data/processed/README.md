# Processed Corpus

Chunked JSONL files produced by `scripts/ingest.py` are written here.

Generated corpus files are ignored by Git. Recreate them from `data/raw/` with:

```bash
python scripts/ingest.py --config configs/default.yaml --output data/processed/processed.jsonl
```
