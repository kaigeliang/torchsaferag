#!/usr/bin/env python
import json
import time
from pathlib import Path
import typer

from torchsaferag.config import load_config
from torchsaferag.evaluation import judge_score

app = typer.Typer(help="Run LLM-as-judge on saved predictions JSONL.")


def load_preds(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


@app.command()
def run(
    config: str = typer.Option("configs/default.yaml", help="Config file (judge settings)"),
    pred_file: str = typer.Option(..., help="Predictions JSONL from eval_generation.py --pred-out"),
    max_samples: int = typer.Option(None, help="Optional cap on number of samples"),
    bootstrap: int = typer.Option(0, help="If >0, run bootstrap for mean CI with given samples"),
):
    cfg = load_config(config)
    judge_model = cfg["evaluation"]["judge_model"]
    api_key_env = cfg["evaluation"].get("judge_api_key_env", "OPENAI_API_KEY")
    api_base_env = cfg["evaluation"].get("judge_api_base_env", "OPENAI_API_BASE")
    if not judge_model:
        raise RuntimeError("judge_model not set in config")

    scores = []
    for i, rec in enumerate(load_preds(Path(pred_file))):
        if max_samples and i >= max_samples:
            break
        try:
            s = judge_score(
                question=rec["question"],
                answer=rec["prediction"],
                context_chunks=rec.get("cited_texts", []),
                model=judge_model,
                api_key_env=api_key_env,
                api_base_env=api_base_env,
            )
            scores.append(s)
        except Exception as e:
            typer.echo(f"[warn] judge failed on {rec.get('id')}: {e}")
        time.sleep(0.5)  # gentle pacing

    if scores:
        avg = sum(scores) / len(scores)
        typer.echo(f"Judge Score (0-5): {avg:.4f} over {len(scores)} samples")
        if bootstrap and len(scores) > 1:
            import random
            means = []
            for _ in range(bootstrap):
                samp = [scores[random.randrange(len(scores))] for _ in range(len(scores))]
                means.append(sum(samp)/len(samp))
            means.sort()
            lo = means[int(0.025*bootstrap)]
            hi = means[int(0.975*bootstrap)]
            typer.echo(f"Bootstrap 95% CI: [{lo:.4f}, {hi:.4f}] (n={bootstrap})")
    else:
        typer.echo("No scores produced.")


if __name__ == "__main__":
    app()
