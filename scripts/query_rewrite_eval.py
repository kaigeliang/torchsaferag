#!/usr/bin/env python
import json
from pathlib import Path
import typer

import faiss
from sentence_transformers import SentenceTransformer

from torchsaferag.config import load_config
from torchsaferag.retrieval import load_catalog, load_bm25, search, hybrid_search_with_vec
from torchsaferag.query_rewrite import RewriteConfig, load_llm, condense_question, hyde

app = typer.Typer(help="Evaluate query rewriting strategies (original vs condense vs HyDE).")


def load_qa(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def recall_at_k(hit_ids, gold, k):
    top = hit_ids[:k]
    return 1.0 if any(h in gold for h in top) else 0.0


@app.command()
def run(
    config: str = typer.Option("configs/default.yaml"),
    qa_file: str = typer.Option("eval/qa.jsonl"),
    catalog: str = typer.Option("embeddings/catalog.parquet"),
    faiss_index_path: str = typer.Option("indexes/faiss_flat.index"),
    bm25_path: str = typer.Option(None),
    k: int = typer.Option(10),
):
    cfg = load_config(config)
    ids, texts, meta = load_catalog(Path(catalog))
    faiss_index = faiss.read_index(str(faiss_index_path))
    bm25 = load_bm25(Path(bm25_path)) if bm25_path else None
    encoder = SentenceTransformer(cfg["embedding"]["model_name"])
    alpha = cfg["retrieval"]["hybrid"]["alpha"]

    rw_cfg = RewriteConfig(
        provider=cfg["generation"].get("provider", "siliconflow"),
        api_base=cfg["generation"].get("api_base"),
        api_key_env=cfg["generation"].get("api_key_env", "SILICONFLOW_API_KEY"),
        model_name=cfg["generation"].get("model_name"),
        temperature=0.3,
        max_new_tokens=128,
    )
    llm = load_llm(rw_cfg)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    stats = {"orig": [], "condense": [], "hyde": []}

    for sample in load_qa(Path(qa_file)):
        gold = sample.get("citation_ground_truth", [])

        # original
        res_orig = search(
            sample["question"],
            bm25=bm25,
            faiss_index=faiss_index,
            catalog_ids=ids,
            catalog_texts=texts,
            alpha=alpha,
            top_k=k,
            query_encoder=encoder,
        )
        stats["orig"].append(recall_at_k([r["id"] for r in res_orig], gold, k))

        # condense
        condensed = condense_question(llm, sample["question"])
        res_cond = search(
            condensed,
            bm25=bm25,
            faiss_index=faiss_index,
            catalog_ids=ids,
            catalog_texts=texts,
            alpha=alpha,
            top_k=k,
            query_encoder=encoder,
        )
        stats["condense"].append(recall_at_k([r["id"] for r in res_cond], gold, k))

        # HyDE
        hypo = hyde(llm, sample["question"])
        hypo_vec = encoder.encode([hypo], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(hypo_vec)
        res_hyde = hybrid_search_with_vec(
            hypo_vec,
            bm25=bm25,
            faiss_index=faiss_index,
            catalog_ids=ids,
            catalog_texts=texts,
            alpha=alpha,
            top_k=k,
            query_text=sample["question"],
        )
        stats["hyde"].append(recall_at_k([r["id"] for r in res_hyde], gold, k))

    typer.echo(f"Recall@{k} original : {avg(stats['orig']):.4f}")
    typer.echo(f"Recall@{k} condense : {avg(stats['condense']):.4f}")
    typer.echo(f"Recall@{k} HyDE     : {avg(stats['hyde']):.4f}")


if __name__ == "__main__":
    app()
