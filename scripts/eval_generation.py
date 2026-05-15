#!/usr/bin/env python
import json
from pathlib import Path
from typing import Optional
import typer

import faiss
from sentence_transformers import SentenceTransformer

from torchsaferag.config import load_config
from torchsaferag.retrieval import load_catalog, load_bm25, search, rerank
from torchsaferag.generation import GenerationConfig, load_llm, generate_answer
from torchsaferag.evaluation import (
    exact_match,
    loose_exact_match,
    f1_score,
    hallucination_rate,
    judge_score,
)

app = typer.Typer(help="Evaluate generation metrics (EM, F1, hallucination).")


def load_qa(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            yield json.loads(line)


@app.command()
def run(
    config: str = typer.Option("configs/default.yaml", help="Config file"),
    qa_file: str = typer.Option("eval/qa.jsonl", help="QA JSONL"),
    catalog: str = typer.Option("embeddings/catalog.parquet", help="Catalog parquet path"),
    faiss_index_path: str = typer.Option("indexes/faiss_flat.index", help="FAISS index path"),
    bm25_path: str = typer.Option(None, help="Optional BM25 pickle"),
    k: int = typer.Option(5, help="Top-k retrieval"),
    pred_out: str = typer.Option(None, help="Optional JSONL to save predictions"),
    prompt_style: Optional[str] = typer.Option(
        None,
        help="Prompt style override: strict or vanilla. Defaults to config.",
    ),
    loose_em: bool = typer.Option(
        False,
        help="Use loose exact match (normalized + Jaccard>=0.8 or substring) to get partial credit.",
    ),
):
    cfg = load_config(config)
    ids, texts, meta = load_catalog(Path(catalog))
    faiss_index = faiss.read_index(str(faiss_index_path))
    bm25 = load_bm25(Path(bm25_path)) if bm25_path else None
    encoder = SentenceTransformer(cfg["embedding"]["model_name"])
    alpha = cfg["retrieval"]["hybrid"]["alpha"]

    gen_cfg = GenerationConfig(
        provider=cfg["generation"]["provider"],
        api_base=cfg["generation"]["api_base"],
        api_key_env=cfg["generation"]["api_key_env"],
        model_name=cfg["generation"]["model_name"],
        context_window=cfg["generation"]["context_window"],
        temperature=cfg["generation"]["temperature"],
        top_k=cfg["generation"]["top_k"],
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        prompt_style=prompt_style or cfg["generation"].get("prompt_style", "strict"),
    )
    llm = load_llm(gen_cfg)

    em_scores = []
    f1_scores = []
    hallucinations = []
    judge_scores = []
    judge_severity = []
    judge_faithfulness = []
    records = []

    for sample in load_qa(Path(qa_file)):
        if k > 0:
            results = search(
                sample["question"],
                bm25=bm25,
                faiss_index=faiss_index,
                catalog_ids=ids,
                catalog_texts=texts,
                alpha=alpha,
                top_k=k,
                query_encoder=encoder,
            )
            if cfg["retrieval"]["rerank"]["enabled"]:
                results = rerank(
                    sample["question"],
                    candidates=results,
                    model_name=cfg["retrieval"]["rerank"]["model_name"],
                    top_n=cfg["retrieval"]["rerank"]["top_n"],
                )
        else:
            # closed-book: no context
            results = []
        answer = generate_answer(llm, sample["question"], results, gen_cfg)
        gold = sample["ground_truth"]
        em_fn = loose_exact_match if loose_em else exact_match
        em_scores.append(em_fn(answer, gold))
        f1_scores.append(f1_score(answer, gold))
        cited_texts = [r["text"] for r in results] if results else []
        hallucinations.append(
            hallucination_rate(
                answer,
                cited_texts,
                sentence_pattern=cfg["evaluation"]["hallucination"]["sentence_split_regex"],
            )
        )
        if cfg["evaluation"].get("judge_model"):
            try:
                js = judge_score(
                    question=sample["question"],
                    answer=answer,
                    context_chunks=cited_texts,
                    model=cfg["evaluation"]["judge_model"],
                    api_key_env=cfg["evaluation"].get("judge_api_key_env", "OPENAI_API_KEY"),
                    api_base_env=cfg["evaluation"].get("judge_api_base_env", "OPENAI_API_BASE"),
                )
                judge_scores.append(js)
                judge_severity.append(1.0 - js / 5.0)
                judge_faithfulness.append(js / 5.0)
            except Exception as e:
                typer.echo(f"[warn] judge failed for '{sample['question'][:40]}...': {e}")
        # collect record
        records.append(
            {
                "id": sample.get("id"),
                "question": sample["question"],
                "ground_truth": gold,
                "prediction": answer,
                "retrieved": [{"id": r["id"], "text": r["text"], "score": r.get("score")} for r in results],
                "cited_texts": cited_texts,
                "difficulty": sample.get("difficulty"),
            }
        )

    def avg(arr):
        return sum(arr) / len(arr) if arr else 0.0

    em_label = "EM_loose" if loose_em else "EM"
    typer.echo(f"{em_label}: {avg(em_scores):.4f}")
    typer.echo(f"F1: {avg(f1_scores):.4f}")
    typer.echo(f"Hallucination Rate: {avg(hallucinations):.4f}")
    if judge_scores:
        typer.echo(f"Judge Score (0-5): {avg(judge_scores):.4f}")
        typer.echo(f"Hallucination Severity (1-score/5): {avg(judge_severity):.4f}")
        typer.echo(f"Faithfulness (score/5): {avg(judge_faithfulness):.4f}")
    if pred_out:
        Path(pred_out).parent.mkdir(parents=True, exist_ok=True)
        with Path(pred_out).open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        typer.echo(f"Saved predictions to {pred_out}")


if __name__ == "__main__":
    app()
