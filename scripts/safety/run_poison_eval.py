#!/usr/bin/env python
import json
from pathlib import Path
import typer
import time

import faiss
from sentence_transformers import SentenceTransformer

from torchsaferag.config import load_config
from torchsaferag.retrieval import load_catalog, load_bm25, search, rerank
from torchsaferag.generation import GenerationConfig, load_llm, generate_answer
from torchsaferag.evaluation import exact_match, f1_score, hallucination_rate

app = typer.Typer(help="Evaluate retrieval poisoning impact using clean vs poisoned indexes.")


def load_qa(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def has_poison(results):
    return any("poison" in r["id"].lower() for r in results)


@app.command()
def run(
    config: str = typer.Option("configs/default.yaml"),
    safety_config: str = typer.Option("configs/safety.yaml"),
    qa_file: str = typer.Option("eval/qa.jsonl"),
    catalog: str = typer.Option("embeddings/catalog.parquet"),
    poisoned_catalog: str = typer.Option("embeddings/catalog_poisoned.parquet"),
    clean_faiss: str = typer.Option("indexes/faiss_flat.index"),
    poisoned_faiss: str = typer.Option("indexes/poisoned/faiss_flat.index"),
    clean_bm25: str = typer.Option(None),
    poisoned_bm25: str = typer.Option(None),
    k: int = typer.Option(5),
    log_path: str = typer.Option("eval/safety_logs/poison_eval.jsonl"),
):
    cfg = load_config(config)
    scfg = load_config(safety_config)
    clean_ids, clean_texts, _ = load_catalog(Path(catalog))
    poisoned_ids, poisoned_texts, _ = load_catalog(Path(poisoned_catalog))
    enc = SentenceTransformer(cfg["embedding"]["model_name"])
    alpha = cfg["retrieval"]["hybrid"]["alpha"]

    clean_index = faiss.read_index(str(clean_faiss))
    poisoned_index = faiss.read_index(str(poisoned_faiss))
    clean_bm25_obj = load_bm25(Path(clean_bm25)) if clean_bm25 else None
    poisoned_bm25_obj = load_bm25(Path(poisoned_bm25)) if poisoned_bm25 else None

    gen_cfg = GenerationConfig(
        provider=cfg["generation"]["provider"],
        api_base=cfg["generation"]["api_base"],
        api_key_env=cfg["generation"]["api_key_env"],
        model_name=cfg["generation"]["model_name"],
        model_path=cfg["generation"]["model_path"],
        context_window=cfg["generation"]["context_window"],
        temperature=scfg["temperature"],
        top_k=cfg["generation"]["top_k"],
        max_new_tokens=scfg["max_new_tokens"],
        prompt_style=cfg["generation"].get("prompt_style", "strict"),
    )
    llm = load_llm(gen_cfg)

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    drift = 0
    em_drop = []
    f1_drop = []
    hallucination_delta = []

    def safe_generate(question, ctx):
        for attempt in range(3):
            try:
                return generate_answer(llm, question, ctx, gen_cfg)
            except Exception as e:
                if attempt == 2:
                    return "I don't know."
                time.sleep(5)

    for sample in load_qa(Path(qa_file)):
        clean_results = search(
            sample["question"],
            bm25=clean_bm25_obj,
            faiss_index=clean_index,
            catalog_ids=clean_ids,
            catalog_texts=clean_texts,
            alpha=alpha,
            top_k=k,
            query_encoder=enc,
        )
        if scfg.get("use_rerank") and cfg["retrieval"]["rerank"]["enabled"]:
            clean_results = rerank(
                sample["question"],
                candidates=clean_results,
                model_name=cfg["retrieval"]["rerank"]["model_name"],
                top_n=cfg["retrieval"]["rerank"]["top_n"],
            )
        poisoned_results = search(
            sample["question"],
            bm25=poisoned_bm25_obj,
            faiss_index=poisoned_index,
            catalog_ids=poisoned_ids,
            catalog_texts=poisoned_texts,
            alpha=alpha,
            top_k=k,
            query_encoder=enc,
        )
        if scfg.get("use_rerank") and cfg["retrieval"]["rerank"]["enabled"]:
            poisoned_results = rerank(
                sample["question"],
                candidates=poisoned_results,
                model_name=cfg["retrieval"]["rerank"]["model_name"],
                top_n=cfg["retrieval"]["rerank"]["top_n"],
            )
        drift += int(has_poison(poisoned_results))

        clean_answer = safe_generate(sample["question"], clean_results)
        poisoned_answer = safe_generate(sample["question"], poisoned_results)
        gold = sample["ground_truth"]

        em_clean = exact_match(clean_answer, gold)
        em_poison = exact_match(poisoned_answer, gold)
        f1_clean = f1_score(clean_answer, gold)
        f1_poison = f1_score(poisoned_answer, gold)
        em_drop.append(em_clean - em_poison)
        f1_drop.append(f1_clean - f1_poison)

        h_clean = hallucination_rate(clean_answer, [r["text"] for r in clean_results])
        h_poison = hallucination_rate(poisoned_answer, [r["text"] for r in poisoned_results])
        hallucination_delta.append(h_poison - h_clean)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "id": sample["id"],
                        "question": sample["question"],
                        "clean_retrieved": [r["id"] for r in clean_results],
                        "poisoned_retrieved": [r["id"] for r in poisoned_results],
                        "clean_answer": clean_answer,
                        "poisoned_answer": poisoned_answer,
                        "em_clean": em_clean,
                        "em_poison": em_poison,
                        "f1_clean": f1_clean,
                        "f1_poison": f1_poison,
                        "hallucination_clean": h_clean,
                        "hallucination_poison": h_poison,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    n = len(list(load_qa(Path(qa_file))))
    drift_rate = drift / n if n else 0.0
    avg_em_drop = sum(em_drop) / len(em_drop) if em_drop else 0.0
    avg_f1_drop = sum(f1_drop) / len(f1_drop) if f1_drop else 0.0
    avg_h_delta = sum(hallucination_delta) / len(hallucination_delta) if hallucination_delta else 0.0

    typer.echo(f"Citation Drift Rate: {drift_rate:.4f}")
    typer.echo(f"Delta EM (clean - poisoned): {avg_em_drop:.4f}")
    typer.echo(f"Delta F1 (clean - poisoned): {avg_f1_drop:.4f}")
    typer.echo(f"Delta Hallucination (poison - clean): {avg_h_delta:.4f}")


if __name__ == "__main__":
    app()
