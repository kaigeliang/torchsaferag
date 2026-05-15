#!/usr/bin/env python
import re
from pathlib import Path
import typer


EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"\+?\d[\d\-\s]{6,}\d")


def redact(text: str) -> str:
    text = EMAIL.sub("[REDACTED_EMAIL]", text)
    text = PHONE.sub("[REDACTED_PHONE]", text)
    return text


def process_file(path: Path, output: Path):
    raw = path.read_text(encoding="utf-8")
    clean = redact(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(clean, encoding="utf-8")


app = typer.Typer(help="Redact basic PII patterns (email, phone) from text/markdown.")


@app.command()
def run(
    input_path: str = typer.Argument(..., help="Input file (.txt/.md)"),
    output_path: str = typer.Argument(..., help="Output redacted file"),
):
    process_file(Path(input_path), Path(output_path))
    typer.echo(f"Redacted file written to {output_path}")


if __name__ == "__main__":
    app()
