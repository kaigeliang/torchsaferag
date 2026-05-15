#!/usr/bin/env python
import typer
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from torchsaferag.config import load_config
from torchsaferag.retrieval import load_catalog, load_bm25, search, rerank
from torchsaferag.generation import GenerationConfig, load_llm, generate_answer

app = typer.Typer(help="Run a single RAG query.")


@app.command()
def run(
    question: str = typer.Argument(..., help="User question"),
    config: str = typer.Option("configs/default.yaml", help="Config file"),
    catalog: str = typer.Option("embeddings/catalog.parquet", help="Parquet catalog path"),
    embeddings: str = typer.Option("embeddings/catalog.npy", help="Embeddings .npy path"),
    faiss_index_path: str = typer.Option("indexes/faiss_flat.index", help="FAISS index path"),
    bm25_path: str = typer.Option("indexes/bm25.pkl", help="Optional BM25 pickle path"),
    top_k: int = typer.Option(5, help="Top k context chunks"),
):
    cfg = load_config(config)
    ids, texts, meta = load_catalog(Path(catalog))

    # load indexes
    faiss_index = faiss.read_index(str(faiss_index_path))
    bm25_candidate = Path(bm25_path) if bm25_path else None
    bm25 = load_bm25(bm25_candidate) if bm25_candidate and bm25_candidate.exists() else None

    alpha = cfg["retrieval"]["hybrid"]["alpha"]
    encoder = SentenceTransformer(cfg["embedding"]["model_name"])
    results = search(
        question,
        bm25=bm25,
        faiss_index=faiss_index,
        catalog_ids=ids,
        catalog_texts=texts,
        alpha=alpha,
        top_k=top_k,
        query_encoder=encoder,
    )
    if cfg["retrieval"]["rerank"]["enabled"]:
        results = rerank(
            question,
            candidates=results,
            model_name=cfg["retrieval"]["rerank"]["model_name"],
            top_n=cfg["retrieval"]["rerank"]["top_n"],
        )

    gen_cfg = GenerationConfig(
        provider=cfg["generation"]["provider"],
        api_base=cfg["generation"]["api_base"],
        api_key_env=cfg["generation"]["api_key_env"],
        model_name=cfg["generation"]["model_name"],
        model_path=cfg["generation"].get("model_path"),
        context_window=cfg["generation"]["context_window"],
        temperature=cfg["generation"]["temperature"],
        top_k=cfg["generation"]["top_k"],
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        prompt_style=cfg["generation"].get("prompt_style", "strict"),
    )
    llm = load_llm(gen_cfg)
    answer = generate_answer(llm, question, results, gen_cfg)
    typer.echo("Answer:\n" + answer.strip())
    typer.echo("\nCitations:")
    for r in results:
        typer.echo(f"- {r['id']} (score={r['score']:.4f})")


if __name__ == "__main__":
    app()
