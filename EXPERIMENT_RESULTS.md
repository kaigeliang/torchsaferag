# Experiment Results - TorchSafeRAG

Date: 2025-12-07

Scope: RAG system over PyTorch documentation and AI safety papers.

Default setup: `BAAI/bge-base-en` embeddings, FAISS flat dense index, optional BM25 hybrid retrieval with alpha 0.5, rerank off, seed 42, and `Qwen/Qwen2.5-7B-Instruct` through SiliconFlow for generation.

## Chunking Comparison

Full QA set: `eval/qa.jsonl` with 80 items.

### Recall/MRR at 10

| Chunking | Recall@10 | MRR@10 | nDCG@10 | Recall@3 | Recall@1 |
| --- | --- | --- | --- | --- | --- |
| Sentence (220/20) | 0.5125 | 0.2995 | 0.3330 | 0.3625 | 0.2000 |
| Structured (section-aware, 220/20) | 0.5125 | 0.2995 | 0.3330 | 0.3625 | 0.2000 |
| Fixed (400/40) | 0.5250 | 0.2913 | 0.3148 | 0.3625 | 0.1750 |

### Recall/MRR at 5

| Chunking | Recall@5 | MRR@5 | nDCG@5 | Recall@3 | Recall@1 |
| --- | --- | --- | --- | --- | --- |
| Sentence (220/20) | 0.4375 | 0.3035 | 0.3262 | 0.3875 | 0.2250 |
| Structured (section-aware, 220/20) | 0.4375 | 0.3035 | 0.3262 | 0.3875 | 0.2250 |
| Fixed (400/40) | 0.4250 | 0.2850 | 0.3041 | 0.3500 | 0.1750 |

## Retriever Comparison

Sentence chunking, full QA set.

| Mode | Alpha | Recall@10 | MRR@10 | Recall@5 | Recall@3 | Recall@1 |
| --- | --- | --- | --- | --- | --- | --- |
| Dense | - | 0.5125 | 0.3142 | 0.4375 | 0.3875 | 0.2125 |
| BM25 | - | 0.4375 | 0.2714 | 0.3500 | 0.3500 | 0.1750 |
| Hybrid | 0.3 | 0.5125 | 0.2995 | 0.4375 | 0.3625 | 0.2000 |
| Hybrid | 0.5 | 0.5125 | 0.2995 | 0.4375 | 0.3625 | 0.2000 |
| Hybrid | 0.7 | 0.5125 | 0.2995 | 0.4375 | 0.3625 | 0.2000 |

### Rerank

Cross-encoder rerank was tested on the first 20 QA samples.

| Setting | Recall@5 | MRR@5 | nDCG@5 | Notes |
| --- | --- | --- | --- | --- |
| Dense, no rerank | 0.7000 | 0.5167 | 0.5643 | `eval/qa_sample20.jsonl` |
| Dense + `BAAI/bge-reranker-base` | 0.7000 | 0.5000 | 0.5512 | CPU batch size 8 |

Rerank did not improve this subset and added CPU latency, so the default config keeps rerank off.

## Generation And Judge

Answers were generated with Qwen through SiliconFlow. Judge scores used `gpt-4o-mini` through an OpenAI-compatible endpoint.

| Mode | k | EM | F1 | Hallucination Rate | Judge Score (0-5) | Severity | Faithfulness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAG | 5 | 0.0000 | 0.1544 | 0.9306 | 3.5875 | 0.2825 | 0.7175 |
| Closed-book | 0 | 0.0000 | 0.0609 | 1.0000 | 1.6875 | 0.6625 | 0.3375 |

Post-hoc loose EM was 0.0500 for RAG k=5 and 0.0125 for closed-book.

## Prompt Comparison

| Mode | Prompt | k | EM | F1 | Hallucination Rate |
| --- | --- | --- | --- | --- | --- |
| RAG | strict | 5 | 0.0000 | 0.1544 | 0.9306 |
| RAG | vanilla | 5 | 0.0000 | 0.1467 | 0.9196 |
| Closed-book | strict | 0 | 0.0000 | 0.0609 | 1.0000 |
| Closed-book | vanilla | 0 | 0.0000 | 0.0638 | 1.0000 |

With judge scoring, strict RAG scored 3.5875 and vanilla RAG scored 3.5312.

## Safety

### Prompt Injection

| Attacks | top_k | rerank | Temperature | Max New Tokens | Attack Success Rate |
| --- | --- | --- | --- | --- | --- |
| 30 template attacks | 5 | off | 0.3 | 200 | 0.9667 |

Attack success rate is a simple heuristic based on `hallucination_rate > 0.5`.

### Retrieval Poisoning

Sample: `eval/qa_sample10.jsonl`.

| QA samples | k | Citation Drift Rate | Delta EM | Delta F1 | Delta Hallucination |
| --- | --- | --- | --- | --- | --- |
| 10 | 5 | 0.0000 | 0.0000 | 0.0011 | 0.0000 |

## Efficiency

Strict prompt, judge off, SiliconFlow Qwen.

| QA samples | k | EM | F1 | Hallucination Rate | Avg Pred Tokens |
| --- | --- | --- | --- | --- | --- |
| 20 | 5 | 0.0000 | 0.2397 | 0.8987 | 27.6 |
| 10 | 10 | 0.0000 | 0.2868 | 0.8467 | 36.5 |

Latency on `eval/qa_sample10.jsonl`:

| k | Wall-clock (s) | EM | F1 | Hallucination Rate |
| --- | --- | --- | --- | --- |
| 5 | 24.10 | 0.0000 | 0.2990 | 0.9433 |
| 10 | 24.34 | 0.0000 | 0.2809 | 0.8350 |

Judge results for the latency runs: k=5 scored 4.80; k=10 scored 4.85.

## Query Rewrite

`eval/qa_sample10.jsonl`, sentence index, k=5.

| Method | Recall@5 |
| --- | --- |
| Original query | 0.7000 |
| Condense rewrite | 0.7000 |
| HyDE | 0.8000 |

HyDE improved Recall@5 by 0.10 on this small sample.

## Privacy And PII

Regex scanning found email-like strings in 4 PDFs: `2023_emnlp_main_849.pdf`, `2025_findings_emnlp_698.pdf`, `2302.12173.pdf`, and `2306.05499.pdf`. No phone numbers matched.

Redacted copies were written to `tmp/redacted_raw/`. The original corpus was not modified. For deployment, rebuild embeddings and indexes from redacted source files.

Sampled retrieval after redaction did not materially degrade:

| Set | Recall@5 | MRR@5 |
| --- | --- | --- |
| `eval/qa_sample20.jsonl` | 0.7000 | 0.5208 |
| `eval/qa_sample10.jsonl` | 0.7000 | 0.5833 |

## Reproduction Commands

```powershell
python scripts/ingest.py --config configs/default.yaml --output data/processed/processed.jsonl
python scripts/embed.py --config configs/default.yaml --input-jsonl data/processed/processed.jsonl --output embeddings/catalog.parquet
python scripts/build_index.py --catalog embeddings/catalog.parquet --embeddings embeddings/catalog.npy --output-dir indexes --index-type flat --build-sparse

python scripts/eval_retrieval.py --config configs/default.yaml --qa-file eval/qa.jsonl --catalog embeddings/catalog.parquet --faiss-index-path indexes/faiss_flat.index --bm25-path indexes/bm25.pkl --k 10 --mode hybrid

python scripts/eval_generation.py --config configs/default.yaml --qa-file eval/qa.jsonl --k 5 --pred-out eval/preds_rag_qwen.jsonl
python scripts/eval_generation.py --config configs/default.yaml --qa-file eval/qa.jsonl --k 0 --pred-out eval/preds_closed_qwen.jsonl

python scripts/eval_judge.py --config configs/default.yaml --preds eval/preds_rag_qwen.jsonl --out eval/judge_rag_qwen.jsonl
python scripts/eval_judge.py --config configs/default.yaml --preds eval/preds_closed_qwen.jsonl --out eval/judge_closed_qwen.jsonl

python scripts/eval_generation.py --config configs/default.yaml --qa-file eval/qa.jsonl --k 5 --prompt-style vanilla --pred-out eval/preds_rag_qwen_vanilla.jsonl

python scripts/query_rewrite_eval.py --config configs/default.yaml --qa-file eval/qa_sample10.jsonl --catalog embeddings/catalog.parquet --faiss-index-path indexes/faiss_flat.index --bm25-path indexes/bm25.pkl --k 5

python scripts/safety/run_prompt_injection.py --config configs/default.yaml --log-path eval/safety_logs/prompt_injection.jsonl
python scripts/create_poison_index.py --config configs/default.yaml --processed data/processed/processed.jsonl --poison-dir data/poison --merged-out data/processed/processed_poisoned.jsonl --catalog-out embeddings/catalog_poisoned.parquet --index-out-dir indexes/poisoned
python scripts/safety/run_poison_eval.py --config configs/default.yaml --qa-file eval/qa_sample10.jsonl --catalog embeddings/catalog.parquet --poisoned-catalog embeddings/catalog_poisoned.parquet --clean-faiss indexes/faiss_flat.index --poisoned-faiss indexes/poisoned/faiss_flat.index --clean-bm25 indexes/bm25.pkl --poisoned-bm25 indexes/poisoned/bm25.pkl --log-path eval/safety_logs/poison_eval_sample10.jsonl
```
