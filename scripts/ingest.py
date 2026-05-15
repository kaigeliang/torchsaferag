#!/usr/bin/env python
import typer
from pathlib import Path
from typing import Optional

from torchsaferag.config import load_config
from torchsaferag.data import ingest_paths

app = typer.Typer(help="Clean and chunk raw documents into JSONL.")


@app.command()
def run(
    config: str = typer.Option("configs/default.yaml", help="Path to config YAML"),
    raw_dir: str = typer.Option(None, help="Override raw directory"),
    output: str = typer.Option("data/processed/processed.jsonl", help="Output JSONL path"),
):
    cfg = load_config(config)
    raw_path = Path(raw_dir) if raw_dir else Path(cfg["paths"]["raw_dir"])
    files = list(raw_path.rglob("*"))
    if not files:
        typer.echo(f"No files found under {raw_path}")
        raise typer.Exit(code=1)
    max_tokens = cfg["chunking"]["max_tokens"]
    overlap = cfg["chunking"]["overlap"]
    method = cfg["chunking"].get("method", "fixed")
    ingest_paths(files, Path(output), max_tokens=max_tokens, overlap=overlap, source_prefix="", method=method)


if __name__ == "__main__":
    app()
