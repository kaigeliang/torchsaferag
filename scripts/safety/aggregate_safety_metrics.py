#!/usr/bin/env python
import json
from pathlib import Path
import typer

app = typer.Typer(help="Aggregate safety log JSONL files to summary metrics.")


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


@app.command()
def prompt_injection(log: str = typer.Argument("eval/safety_logs/prompt_injection.jsonl")):
    records = list(load_jsonl(Path(log)))
    if not records:
        typer.echo("No records found.")
        raise typer.Exit()
    success = sum(1 for r in records if r.get("success"))
    rate = success / len(records)
    avg_h = sum(r.get("hallucination_rate", 0.0) for r in records) / len(records)
    typer.echo(f"Attack Success Rate: {rate:.4f}")
    typer.echo(f"Avg Hallucination Rate: {avg_h:.4f}")


@app.command()
def poison(log: str = typer.Argument("eval/safety_logs/poison_eval.jsonl")):
    records = list(load_jsonl(Path(log)))
    if not records:
        typer.echo("No records found.")
        raise typer.Exit()
    drift = sum(1 for r in records if any("poison" in rid.lower() for rid in r.get("poisoned_retrieved", [])))
    drift_rate = drift / len(records)
    avg_em_drop = sum(r["em_clean"] - r["em_poison"] for r in records) / len(records)
    avg_f1_drop = sum(r["f1_clean"] - r["f1_poison"] for r in records) / len(records)
    avg_h_delta = sum(r["hallucination_poison"] - r["hallucination_clean"] for r in records) / len(records)
    typer.echo(f"Citation Drift Rate: {drift_rate:.4f}")
    typer.echo(f"Delta EM (clean - poisoned): {avg_em_drop:.4f}")
    typer.echo(f"Delta F1 (clean - poisoned): {avg_f1_drop:.4f}")
    typer.echo(f"Delta Hallucination (poison - clean): {avg_h_delta:.4f}")


if __name__ == "__main__":
    app()
