from __future__ import annotations

import re
import os
import json
import string
import requests
from typing import List, Dict, Any, Sequence
from collections import Counter


def split_statements(text: str, pattern: str) -> List[str]:
    """
    Split answer into statements (sentences). Uses the provided regex pattern;
    falls back to period split if the pattern is invalid.
    """
    try:
        parts = re.split(pattern, text)
    except re.error:
        parts = text.split(".")
    return [p.strip() for p in parts if p.strip()]


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def supported(statement: str, cited_chunks: Sequence[str], overlap_threshold: float = 0.5) -> bool:
    """
    Citation-grounded support check per hallucation_rate_definition.md:
    - A statement is supported if its main factual content appears in any cited chunk.
    - Heuristics: substring containment OR token Jaccard >= overlap_threshold (default 0.5).
    """
    if not cited_chunks:
        return False
    st_lower = statement.strip().lower()
    if not st_lower:
        return True  # ignore empty fragments

    def _tokenize_alpha_num(text: str) -> List[str]:
        table = str.maketrans("", "", string.punctuation)
        return [w for w in text.lower().translate(table).split() if len(w) >= 3]

    st_tokens = _tokenize_alpha_num(st_lower)
    st_set = set(st_tokens)
    for chunk in cited_chunks:
        chunk_lower = chunk.lower()
        if st_lower in chunk_lower or chunk_lower in st_lower:
            return True
        chunk_tokens = _tokenize_alpha_num(chunk_lower)
        if not chunk_tokens or not st_set:
            continue
        inter = st_set & set(chunk_tokens)
        union = st_set | set(chunk_tokens)
        jacc = len(inter) / len(union) if union else 0.0
        if jacc >= overlap_threshold:
            return True
    return False


def hallucination_rate(
    answer: str,
    cited_chunk_texts: List[str],
    sentence_pattern: str = r"[.!?\u3002\uff1f\uff01]",
) -> float:
    """
    Citation-grounded hallucination rate:
    - Split answer into statements.
    - Mark a statement hallucinated if no cited chunk supports it.
    - Rate = unsupported / total statements.
    """
    statements = split_statements(answer, sentence_pattern)
    if not statements:
        return 0.0
    unsupported = [s for s in statements if not supported(s, cited_chunk_texts)]
    return len(unsupported) / len(statements)


def exact_match(pred: str, gold: str) -> float:
    return float(pred.strip() == gold.strip())


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _extract_float(text: str, default: float = 0.0) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else default


def judge_score(
    question: str,
    answer: str,
    context_chunks: List[str],
    model: str,
    api_key_env: str = "OPENAI_API_KEY",
    api_base_env: str = "OPENAI_API_BASE",
) -> float:
    """
    LLM-as-a-judge scoring: 0 (unsupported/incorrect) .. 5 (fully supported & correct).
    Uses OpenAI-compatible /chat/completions endpoint via requests.
    """
    # lazy .env load (same behavior as generation.py)
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            if not line or line.strip().startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing judge API key in env {api_key_env}")
    base = os.getenv(api_base_env, "")
    # allow ${ENV} placeholder
    if base.startswith("${") and base.endswith("}"):
        base = os.getenv(base[2:-1], base)
    if not base:
        raise RuntimeError(f"Missing judge API base in env {api_base_env}")
    url = base

    context = "\n".join(f"[ctx_{i+1}] {c}" for i, c in enumerate(context_chunks))
    prompt = (
        "You are an impartial grader. Score the answer from 0 (wrong/unsupported) to 5 (fully correct and supported), "
        "allow half-points. Only reply with the numeric score.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer: {answer}\n\nScore:"
    )
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 5,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return _extract_float(content, default=0.0)


def loose_exact_match(
    pred: str,
    gold: str,
    jaccard_threshold: float = 0.4,
    recall_floor: float = 0.5,
) -> float:
    """
    A more forgiving EM:
    - lowercase, strip punctuation
    - match if one string is a substring of the other, or Jaccard overlap of tokens >= threshold
    """
    def _normalize(s: str) -> str:
        return re.sub(r"[^a-z0-9\s]", " ", s.lower()).strip()

    p_norm = _normalize(pred)
    g_norm = _normalize(gold)
    if not p_norm or not g_norm:
        return 0.0
    if p_norm == g_norm or p_norm in g_norm or g_norm in p_norm:
        return 1.0
    p_tokens = p_norm.split()
    g_tokens = g_norm.split()
    if not p_tokens or not g_tokens:
        return 0.0
    overlap = len(set(p_tokens) & set(g_tokens))
    denom = len(set(p_tokens) | set(g_tokens))
    jacc = overlap / denom if denom else 0.0
    recall = overlap / len(set(g_tokens))
    return 1.0 if (jacc >= jaccard_threshold and recall >= recall_floor) else 0.0
