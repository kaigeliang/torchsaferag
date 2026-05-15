from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from pathlib import Path
import os


@dataclass
class RewriteConfig:
    provider: str = "siliconflow"  # or "openai"
    api_base: str | None = None
    api_key_env: str = "SILICONFLOW_API_KEY"
    model_name: str | None = None
    temperature: float = 0.3
    max_new_tokens: int = 128


def load_llm(cfg: RewriteConfig) -> Callable[..., str]:
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

    api_key = os.getenv(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key; set env var {cfg.api_key_env}")
    base = cfg.api_base
    if base and base.startswith("${") and base.endswith("}"):
        base = os.getenv(base[2:-1], base)
    if not base:
        base = os.getenv("OPENAI_API_BASE" if cfg.provider.lower()=="openai" else "SILICONFLOW_API_BASE")
    model_name = cfg.model_name
    provider = cfg.provider.lower()

    def call_api(prompt: str, max_tokens: int, temperature: float) -> str:
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
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    return call_api


def condense_question(llm: Callable[..., str], question: str) -> str:
    prompt = (
        "Rewrite the user question to a concise, self-contained query with full context. "
        "Do not add new facts.\n\nOriginal:\n" + question + "\n\nCondensed:"
    )
    out = llm(prompt=prompt, max_tokens=128, temperature=0.3)
    return out.strip()


def hyde(llm: Callable[..., str], question: str) -> str:
    prompt = (
        "Write a short factual paragraph that would accurately answer the following question. "
        "Stay general; do not cite sources.\n\nQuestion:\n" + question + "\n\nPassage:"
    )
    out = llm(prompt=prompt, max_tokens=180, temperature=0.7)
    return out.strip()
