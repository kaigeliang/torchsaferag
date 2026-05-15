#!/usr/bin/env python
"""
Score pre-generated predictions with LLM-as-Judge + hallucination rate.
Inputs: predictions JSONL from eval_generation.py (--pred-out).
Outputs: aggregated metrics and per-sample log (optional).
"""
import json
import typer
from pathlib import Path
from typing import List

from torchsaferag.evaluation import hallucination_rate, judge_score
from torchsaferag.config import load_config


app = typer.Typer(help="Judge existing prediction files (no re-generation).")


def load_preds(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


@app.command()
def run(
    config: str = typer.Option("configs/default.yaml", help="Config with judge settings"),
    preds: str = typer.Option(..., help="Predictions JSONL (from eval_generation --pred-out)"),
    out: str = typer.Option(None, help="Optional JSONL to write judged records"),
):
    cfg = load_config(config)
    judge_model = cfg["evaluation"].get("judge_model")
    if not judge_model:
        raise typer.Exit("Judge model is not set in config.")

    api_key_env = cfg["evaluation"].get("judge_api_key_env", "OPENAI_API_KEY")
    api_base_env = cfg["evaluation"].get("judge_api_base_env", "OPENAI_API_BASE")
    split_pattern = cfg["evaluation"]["hallucination"]["sentence_split_regex"]

    preds_path = Path(preds)
    out_path = Path(out) if out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    h_rates: List[float] = []
    j_scores: List[float] = []
    j_sev: List[float] = []
    j_faith: List[float] = []

    judged_records = []

    for rec in load_preds(preds_path):
        answer = rec.get("prediction", "")
        question = rec.get("question", "")
        cited_texts = rec.get("cited_texts") or [r.get("text", "") for r in rec.get("retrieved", [])]

        h = hallucination_rate(answer, cited_texts, sentence_pattern=split_pattern)
        h_rates.append(h)

        try:
            js = judge_score(
                question=question,
                answer=answer,
                context_chunks=cited_texts,
                model=judge_model,
                api_key_env=api_key_env,
                api_base_env=api_base_env,
            )
        except Exception as e:
            typer.echo(f"[warn] judge failed for {rec.get('id','?')}: {e}")
            js = 0.0
        j_scores.append(js)
        j_sev.append(1.0 - js / 5.0)
        j_faith.append(js / 5.0)

        if out_path:
            rec["hallucination_rate"] = h
            rec["judge_score"] = js
            rec["hallucination_severity"] = 1.0 - js / 5.0
            rec["faithfulness"] = js / 5.0
            judged_records.append(rec)

    def avg(arr):
        return sum(arr) / len(arr) if arr else 0.0

    typer.echo(f"Samples: {len(h_rates)}")
    typer.echo(f"Hallucination Rate: {avg(h_rates):.4f}")
    typer.echo(f"Judge Score (0-5): {avg(j_scores):.4f}")
    typer.echo(f"Hallucination Severity (1-score/5): {avg(j_sev):.4f}")
    typer.echo(f"Faithfulness (score/5): {avg(j_faith):.4f}")

    if out_path:
        with out_path.open("w", encoding="utf-8") as f:
            for jr in judged_records:
                f.write(json.dumps(jr, ensure_ascii=False) + "\n")
        typer.echo(f"Saved judged records to {out_path}")


if __name__ == "__main__":
    app()
