# Embeddings

Embedding catalogs (`.parquet`) and dense vectors (`.npy`) are generated artifacts.
They are ignored by Git and can be rebuilt with:

```bash
python scripts/embed.py --config configs/default.yaml --input-jsonl data/processed/processed.jsonl --output embeddings/catalog.parquet
```
