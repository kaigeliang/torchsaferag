#!/usr/bin/env python
import typer
from pathlib import Path

from torchsaferag.retrieval import build_faiss_index, build_bm25, load_catalog

app = typer.Typer(help="Build FAISS and BM25 indexes from embeddings/catalog.")


@app.command()
def run(
    catalog: str = typer.Option("embeddings/catalog.parquet", help="Catalog parquet path"),
    embeddings: str = typer.Option("embeddings/catalog.npy", help="Embeddings .npy path"),
    output_dir: str = typer.Option("indexes", help="Output index directory"),
    index_type: str = typer.Option("flat", help="FAISS index type: flat or ivf"),
    build_sparse: bool = typer.Option(True, help="Also build BM25 pickle"),
):
    ids, texts, meta = load_catalog(Path(catalog))
    emb_path = Path(embeddings)
    out_dir = Path(output_dir)
    build_faiss_index(emb_path, out_dir, index_type=index_type)
    if build_sparse:
        bm25_path = out_dir / "bm25.pkl"
        build_bm25(texts, bm25_path)


if __name__ == "__main__":
    app()
