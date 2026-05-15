from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_chunks(jsonl_path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def embed_corpus(
    jsonl_path: Path,
    model_name: str,
    batch_size: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = load_chunks(jsonl_path)
    texts = [r["text"] for r in records]
    model = SentenceTransformer(model_name)

    vectors: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="embedding"):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        vectors.append(emb)

    mat = np.vstack(vectors)
    df = pd.DataFrame(
        {
            "id": [r["id"] for r in records],
            "text": texts,
            "metadata": [r.get("metadata", {}) for r in records],
        }
    )
    # store embeddings in .npy and catalog in parquet for convenience
    np.save(output_path.with_suffix(".npy"), mat)
    df.to_parquet(output_path, index=False)
    print(f"Saved embeddings to {output_path.with_suffix('.npy')} and catalog to {output_path}")
