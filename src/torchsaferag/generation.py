from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Any
import os


@dataclass
class GenerationConfig:
    # remote provider
    provider: str = "siliconflow"  # options: siliconflow, openai
    api_base: str | None = None
    api_key_env: str = "SILICONFLOW_API_KEY"
    model_name: str | None = None
    # legacy/local field (ignored for siliconflow)
    model_path: str | None = None
    # shared generation params
    context_window: int = 4096
    temperature: float = 0.7
    top_k: int = 40
    max_new_tokens: int = 256
    prompt_style: str = "strict"


PROMPT_STYLES = {
    # vanilla: more free-form, still grounded
    "vanilla": (
        "You are a helpful assistant. Use the provided context to answer. "
        "If the answer is missing, say \"I don't know\". "
        "Cite supporting chunks as [chunk_i]."
    ),
    # strict: terse, citation-first
    "strict": (
        "You are a concise, factual assistant. Use ONLY the provided context. "
        "If the answer is not fully supported, say \"I don't know.\" "
        "Reply in ONE short sentence. Include citations as [chunk_i]. No extra words."
    ),
}

# default style
SYSTEM_RULES = PROMPT_STYLES["strict"]


def format_prompt(question: str, context_chunks: List[Dict[str, Any]], rules: str) -> str:
    context_lines = []
    for idx, chunk in enumerate(context_chunks, start=1):
        context_lines.append(f"[chunk_{idx}] {chunk['text']}")
    context_block = "\n".join(context_lines)
    return f"{rules}\n\nQuestion:\n{question}\n\nContext:\n{context_block}\n\nAnswer with citations."


def generate_answer(
    llm: Callable[..., str],
    question: str,
    context_chunks: List[Dict[str, Any]],
    cfg: GenerationConfig,
) -> str:
    # select prompt style
    style = cfg.prompt_style or "strict"
    rules = PROMPT_STYLES.get(style, PROMPT_STYLES["strict"])
    prompt = format_prompt(question, context_chunks, rules)
    resp = llm(
        prompt=prompt,
        max_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
    )
    # siliconflow returns string directly
    if isinstance(resp, str):
        return resp
    # fallback: OpenAI style dict
    if isinstance(resp, dict) and "choices" in resp:
        return resp["choices"][0].get("message", {}).get("content", resp["choices"][0].get("text", ""))
    return str(resp)


def load_llm(cfg: GenerationConfig) -> Callable[..., str]:
    import requests

    # load .env if present
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line or line.strip().startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    provider = cfg.provider.lower()
    api_key = os.getenv(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key; set env var {cfg.api_key_env}")
    base = cfg.api_base
    # allow ${ENV_VAR} placeholder
    if base and base.startswith("${") and base.endswith("}"):
        base = os.getenv(base[2:-1], base)
    if not base:
        # strictly read from env, no defaults
        base = os.getenv("OPENAI_API_BASE" if provider == "openai" else "SILICONFLOW_API_BASE")
    model_name = cfg.model_name or os.getenv("OPENAI_MODEL_NAME")

    if provider == "siliconflow":
        def call_api(prompt: str, max_tokens: int, temperature: float):
            url = base
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            r = requests.post(url, json=payload, headers=headers, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        return call_api

    if provider == "openai":
        def call_api(prompt: str, max_tokens: int, temperature: float):
            url = base
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            r = requests.post(url, json=payload, headers=headers, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        return call_api

    raise RuntimeError(f"Unsupported provider: {cfg.provider}")
