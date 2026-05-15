# Indexes

FAISS and BM25 indexes are generated artifacts. They are ignored by Git and can be
rebuilt after embedding:

```bash
python scripts/build_index.py --catalog embeddings/catalog.parquet --embeddings embeddings/catalog.npy --output-dir indexes --index-type flat --build-sparse
```
