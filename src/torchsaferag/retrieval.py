from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from functools import lru_cache


def load_catalog(parquet_path: Path) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    return df["id"].tolist(), df["text"].tolist(), df["metadata"].tolist()


def build_faiss_index(embeddings_npy: Path, output_dir: Path, index_type: str = "flat") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    vecs = np.load(embeddings_npy)
    dim = vecs.shape[1]
    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    else:
        nlist = min(100, max(1, vecs.shape[0] // 50))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vecs)
    faiss.normalize_L2(vecs)
    index.add(vecs)
    out_path = output_dir / f"faiss_{index_type}.index"
    faiss.write_index(index, str(out_path))
    print(f"FAISS index saved to {out_path}")
    return out_path


def build_bm25(corpus_texts: List[str], output_path: Path) -> Path:
    tokenized = [t.split() for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        import pickle

        pickle.dump(bm25, f)
    print(f"BM25 saved to {output_path}")
    return output_path


def load_bm25(path: Path) -> BM25Okapi:
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)


def search(
    query: str,
    bm25: BM25Okapi | None,
    faiss_index: faiss.Index,
    catalog_ids: List[str],
    catalog_texts: List[str],
    alpha: float = 0.5,
    top_k: int = 5,
    query_encoder: SentenceTransformer | None = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: dense (FAISS) + optional BM25; returns list with id, text, score.
    """
    dense_scores = []
    if query_encoder is None:
        raise ValueError("query_encoder is required for dense search")
    qvec = query_encoder.encode([query], convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(qvec)
    scores, idxs = faiss_index.search(qvec, top_k)
    for score, idx in zip(scores[0], idxs[0]):
        dense_scores.append((int(idx), float(score)))

    sparse_scores = {}
    if bm25:
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        for i, s in enumerate(bm25_scores):
            sparse_scores[i] = float(s)

    # combine
    results = []
    for idx, d_score in dense_scores:
        s_score = sparse_scores.get(idx, 0.0)
        hybrid = alpha * d_score + (1 - alpha) * s_score
        results.append(
            {"id": catalog_ids[idx], "text": catalog_texts[idx], "score": hybrid, "dense": d_score, "sparse": s_score}
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def encode_hypothetical(
    query: str,
    generator,
    query_encoder: SentenceTransformer,
    hyde_prompt: str = "Write a concise factual passage that would answer:",
) -> np.ndarray:
    """
    HyDE: generate hypothetical answer text, then embed it for retrieval.
    The `generator` is expected to have a `generate(query)->str` callable.
    """
    hypo = generator.generate(f"{hyde_prompt} {query}")
    vec = query_encoder.encode([hypo], convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(vec)
    return vec


def hybrid_search_with_vec(
    query_vec: np.ndarray,
    bm25: Optional[BM25Okapi],
    faiss_index: faiss.Index,
    catalog_ids: List[str],
    catalog_texts: List[str],
    alpha: float = 0.5,
    top_k: int = 5,
    query_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    dense_scores = []
    scores, idxs = faiss_index.search(query_vec, top_k)
    for score, idx in zip(scores[0], idxs[0]):
        dense_scores.append((int(idx), float(score)))

    sparse_scores = {}
    if bm25 and query_text:
        tokenized_query = query_text.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        for i, s in enumerate(bm25_scores):
            sparse_scores[i] = float(s)

    results = []
    for idx, d_score in dense_scores:
        s_score = sparse_scores.get(idx, 0.0)
        hybrid = alpha * d_score + (1 - alpha) * s_score
        results.append(
            {"id": catalog_ids[idx], "text": catalog_texts[idx], "score": hybrid, "dense": d_score, "sparse": s_score}
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    model_name: str,
    top_n: int,
) -> List[Dict[str, Any]]:
    """
    Rerank candidate chunks using a cross-encoder.
    """
    if not candidates:
        return []

    @lru_cache(maxsize=2)
    def _load_cross(name: str):
        # force CPU to avoid GPU OOM on small cards
        return CrossEncoder(name, device="cpu")

    cross = _load_cross(model_name)
    pairs = [[query, c["text"]] for c in candidates]
    # small batch to avoid OOM on limited VRAM
    scores = cross.predict(pairs, batch_size=8, show_progress_bar=False)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_n]
