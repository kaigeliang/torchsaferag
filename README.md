# TorchSafeRAG

Safety-aware retrieval-augmented generation over PyTorch documentation and AI safety papers.

This project builds a compact RAG pipeline with document ingestion, BGE embeddings, FAISS dense search, optional BM25 hybrid retrieval, OpenAI-compatible generation, retrieval/generation evaluation, and safety probes for prompt injection and retrieval poisoning.

## Features

- PDF, HTML, Markdown, and text ingestion with fixed or sentence-aware chunking.
- Dense retrieval with FAISS plus optional BM25 hybrid scoring and cross-encoder reranking.
- OpenAI-compatible generation through SiliconFlow or OpenAI-style chat endpoints.
- Retrieval metrics: Recall@k, MRR@k, and nDCG@k.
- Generation metrics: EM, loose EM, F1, citation-grounded hallucination rate, and LLM-as-judge scoring.
- Safety evaluations for prompt injection robustness and poisoned-index drift.

## Repository Layout

- `configs/` experiment and model configuration files.
- `src/torchsaferag/` reusable package code.
- `scripts/` CLI entry points for ingestion, indexing, querying, evaluation, plotting, and safety tests.
- `eval/` QA sets kept in Git; generated prediction and judge logs are ignored.
- `figures/` plots used by the experiment summary.
- `data/`, `embeddings/`, and `indexes/` contain generated or externally sourced artifacts locally. Heavy files in those folders are ignored by Git and can be rebuilt.
- `EXPERIMENT_RESULTS.md` summarizes the main experiment outputs.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Copy the example environment file and fill in real keys:

```powershell
Copy-Item .env.example .env
```

Alternatively, export variables directly in PowerShell:

```powershell
$env:SILICONFLOW_API_KEY = "..."
$env:OPENAI_API_KEY = "..."
$env:OPENAI_API_BASE = "https://api.openai.com/v1/chat/completions"
```

Defaults live in `configs/default.yaml`: seed 42, sentence chunks of 220 words with 20-word overlap, hybrid alpha 0.5, `Qwen/Qwen2.5-7B-Instruct`, strict prompting, and `gpt-4o-mini` for judging.

## Rebuild Artifacts

The generated corpus, embeddings, and indexes are not committed. To reproduce them from a clean checkout, place source PDFs/HTML files under `data/raw/`, then run:

```powershell
python scripts/ingest.py --config configs/default.yaml --output data/processed/processed.jsonl
python scripts/embed.py --config configs/default.yaml --input-jsonl data/processed/processed.jsonl --output embeddings/catalog.parquet
python scripts/build_index.py --catalog embeddings/catalog.parquet --embeddings embeddings/catalog.npy --output-dir indexes --index-type flat --build-sparse
```

## Run a Query

```powershell
python scripts/query.py "How do I move a tensor to CUDA?"
```

## Evaluation

Retrieval:

```powershell
python scripts/eval_retrieval.py --config configs/default.yaml --qa-file eval/qa.jsonl --catalog embeddings/catalog.parquet --faiss-index-path indexes/faiss_flat.index --bm25-path indexes/bm25.pkl --mode hybrid --k 10
```

Generation:

```powershell
python scripts/eval_generation.py --config configs/default.yaml --qa-file eval/qa.jsonl --k 5 --pred-out eval/preds_rag_qwen.jsonl
python scripts/eval_generation.py --config configs/default.yaml --qa-file eval/qa.jsonl --k 0 --pred-out eval/preds_closed_qwen.jsonl
```

LLM-as-judge:

```powershell
python scripts/eval_judge.py --config configs/default.yaml --preds eval/preds_rag_qwen.jsonl --out eval/judge_rag_qwen.jsonl
python scripts/eval_judge.py --config configs/default.yaml --preds eval/preds_closed_qwen.jsonl --out eval/judge_closed_qwen.jsonl
```

Prompt ablation:

```powershell
python scripts/eval_generation.py --config configs/default.yaml --qa-file eval/qa.jsonl --k 5 --prompt-style vanilla --pred-out eval/preds_rag_qwen_vanilla.jsonl
```

Safety probes:

```powershell
python scripts/safety/run_prompt_injection.py --config configs/default.yaml --log-path eval/safety_logs/prompt_injection.jsonl

python scripts/create_poison_index.py --config configs/default.yaml --processed data/processed/processed.jsonl --poison-dir data/poison --merged-out data/processed/processed_poisoned.jsonl --catalog-out embeddings/catalog_poisoned.parquet --index-out-dir indexes/poisoned

python scripts/safety/run_poison_eval.py --config configs/default.yaml --qa-file eval/qa_sample10.jsonl --catalog embeddings/catalog.parquet --poisoned-catalog embeddings/catalog_poisoned.parquet --clean-faiss indexes/faiss_flat.index --poisoned-faiss indexes/poisoned/faiss_flat.index --clean-bm25 indexes/bm25.pkl --poisoned-bm25 indexes/poisoned/bm25.pkl --log-path eval/safety_logs/poison_eval_sample10.jsonl
```

## Notes

- Do not commit `.env` or real API keys.
- `data/raw/`, `data/processed/`, `embeddings/`, `indexes/`, and generated eval logs are ignored to keep the GitHub repository lean.
- Use `EXPERIMENT_RESULTS.md` for the summarized metrics and `figures/` for plotted results.
