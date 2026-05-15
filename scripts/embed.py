#!/usr/bin/env python
import typer
from pathlib import Path

from torchsaferag.config import load_config
from torchsaferag.embedding import embed_corpus

app = typer.Typer(help="Embed processed chunks and store catalog + vectors.")


@app.command()
def run(
    config: str = typer.Option("configs/default.yaml", help="Path to config YAML"),
    input_jsonl: str = typer.Option("data/processed/processed.jsonl", help="Chunked corpus JSONL"),
    output: str = typer.Option("embeddings/catalog.parquet", help="Output catalog parquet path"),
):
    cfg = load_config(config)
    model_name = cfg["embedding"]["model_name"]
    batch_size = cfg["embedding"]["batch_size"]
    embed_corpus(Path(input_jsonl), model_name=model_name, batch_size=batch_size, output_path=Path(output))


if __name__ == "__main__":
    app()
