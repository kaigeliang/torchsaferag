#!/usr/bin/env python
import json
from pathlib import Path
import typer

import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

from torchsaferag.config import load_config
from torchsaferag.retrieval import load_catalog, load_bm25, search, rerank

app = typer.Typer(help="Evaluate retrieval (Recall@k, MRR) on QA set.")


def load_qa(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def recall_at_k(hits, gold, k, all_citations: bool = False):
    top = hits[:k]
    if not gold:
        return 0.0
    if all_citations:
        return 1.0 if all(g in top for g in gold) else 0.0
    return 1.0 if any(h in gold for h in top) else 0.0


def mrr_at_k(hits, gold, k):
    top = hits[:k]
    for rank, h in enumerate(top, start=1):
        if h in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(hits, gold, k):
    """Binary relevance nDCG."""
    top = hits[:k]
    dcg = 0.0
    for i, h in enumerate(top):
        if h in gold:
            dcg += 1.0 / (np.log2(i + 2))
    # ideal DCG
    ideal_rels = [1.0] * min(len(gold), k)
    idcg = sum(rel / (np.log2(i + 2)) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


@app.command()
def run(
    config: str = typer.Option("configs/default.yaml", help="Config file"),
    qa_file: str = typer.Option("eval/qa.jsonl", help="QA JSONL"),
    catalog: str = typer.Option("embeddings/catalog.parquet", help="Catalog parquet path"),
    faiss_index_path: str = typer.Option("indexes/faiss_flat.index", help="FAISS index path"),
    bm25_path: str = typer.Option(None, help="Optional BM25 pickle"),
    k: int = typer.Option(10, help="Top-k to evaluate"),
    mode: str = typer.Option(
        "hybrid",
        help="Retrieval mode: dense | bm25 | hybrid",
        case_sensitive=False,
    ),
    alpha: float = typer.Option(None, help="Hybrid alpha override (0-1); default uses config"),
    all_citations: bool = typer.Option(
        False, help="Require hitting all citation_ground_truth ids to count as correct"
    ),
):
    cfg = load_config(config)
    ids, texts, meta = load_catalog(Path(catalog))
    faiss_index = faiss.read_index(str(faiss_index_path))
    bm25 = load_bm25(Path(bm25_path)) if bm25_path else None
    encoder = SentenceTransformer(cfg["embedding"]["model_name"])
    alpha_val = cfg["retrieval"]["hybrid"]["alpha"] if alpha is None else alpha
    mode = mode.lower()

    recalls = []
    mrrs = []
    recalls1 = []
    mrrs1 = []
    recalls3 = []
    mrrs3 = []
    ndcgs = []
    for sample in load_qa(Path(qa_file)):
        gold = sample.get("citation_ground_truth", [])
        question = sample["question"]

        if mode == "dense":
            # dense only
            qvec = encoder.encode([question], convert_to_numpy=True, show_progress_bar=False)
            faiss.normalize_L2(qvec)
            scores, idxs = faiss_index.search(qvec, k)
            results = [
                {"id": ids[int(idx)], "text": texts[int(idx)], "score": float(score)}
                for score, idx in zip(scores[0], idxs[0])
            ]
        elif mode == "bm25":
            if bm25 is None:
                raise RuntimeError("BM25 index not provided")
            tokenized_query = question.split()
            bm25_scores = bm25.get_scores(tokenized_query)
            top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
            results = [
                {"id": ids[i], "text": texts[i], "score": float(bm25_scores[i])}
                for i in top_idx
            ]
        else:  # hybrid
            results = search(
                question,
                bm25=bm25,
                faiss_index=faiss_index,
                catalog_ids=ids,
                catalog_texts=texts,
                alpha=alpha_val,
                top_k=k,
                query_encoder=encoder,
            )
        if cfg["retrieval"]["rerank"]["enabled"]:
            results = rerank(
                question,
                candidates=results,
                model_name=cfg["retrieval"]["rerank"]["model_name"],
                top_n=cfg["retrieval"]["rerank"]["top_n"],
            )
        hit_ids = [r["id"] for r in results]
        recalls.append(recall_at_k(hit_ids, gold, k, all_citations))
        mrrs.append(mrr_at_k(hit_ids, gold, k))
        recalls1.append(recall_at_k(hit_ids, gold, 1, all_citations))
        mrrs1.append(mrr_at_k(hit_ids, gold, 1))
        recalls3.append(recall_at_k(hit_ids, gold, min(3, k), all_citations))
        mrrs3.append(mrr_at_k(hit_ids, gold, min(3, k)))
        ndcgs.append(ndcg_at_k(hit_ids, gold, k))

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0
    avg_recall1 = sum(recalls1) / len(recalls1) if recalls1 else 0.0
    avg_mrr1 = sum(mrrs1) / len(mrrs1) if mrrs1 else 0.0
    avg_recall3 = sum(recalls3) / len(recalls3) if recalls3 else 0.0
    avg_mrr3 = sum(mrrs3) / len(mrrs3) if mrrs3 else 0.0
    avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0

    typer.echo(f"Recall@{k}: {avg_recall:.4f}")
    typer.echo(f"MRR@{k}: {avg_mrr:.4f}")
    typer.echo(f"nDCG@{k}: {avg_ndcg:.4f}")
    typer.echo(f"Recall@1: {avg_recall1:.4f} | MRR@1: {avg_mrr1:.4f}")
    if k >= 3:
        typer.echo(f"Recall@3: {avg_recall3:.4f} | MRR@3: {avg_mrr3:.4f}")


if __name__ == "__main__":
    app()
