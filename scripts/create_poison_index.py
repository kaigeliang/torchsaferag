#!/usr/bin/env python
import json
from pathlib import Path
import typer

from torchsaferag.config import load_config
from torchsaferag.data import ingest_paths
from torchsaferag.embedding import embed_corpus
from torchsaferag.retrieval import build_faiss_index, build_bm25, load_catalog

app = typer.Typer(help="Create poisoned index by merging processed corpus with poison chunks.")


@app.command()
def run(
    config: str = typer.Option("configs/default.yaml"),
    processed: str = typer.Option("data/processed/processed.jsonl"),
    poison_dir: str = typer.Option("data/poison"),
    merged_out: str = typer.Option("data/processed/processed_poisoned.jsonl"),
    catalog_out: str = typer.Option("embeddings/catalog_poisoned.parquet"),
    index_out_dir: str = typer.Option("indexes/poisoned"),
    index_type: str = typer.Option("flat"),
    build_sparse: bool = typer.Option(True),
):
    cfg = load_config(config)
    # Merge processed + poison
    merged_path = Path(merged_out)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as out:
        with Path(processed).open("r", encoding="utf-8") as f:
            for line in f:
                out.write(line)
        poison_files = list(Path(poison_dir).glob("*.jsonl"))
        for pf in poison_files:
            with pf.open("r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)

    # Embed merged
    embed_corpus(
        merged_path,
        model_name=cfg["embedding"]["model_name"],
        batch_size=cfg["embedding"]["batch_size"],
        output_path=Path(catalog_out),
    )
    ids, texts, meta = load_catalog(Path(catalog_out))
    emb_path = Path(catalog_out).with_suffix(".npy")
    out_dir = Path(index_out_dir)
    build_faiss_index(emb_path, out_dir, index_type=index_type)
    if build_sparse:
        build_bm25(texts, out_dir / "bm25.pkl")

    typer.echo(f"Poisoned index written to {index_out_dir}")


if __name__ == "__main__":
    app()
