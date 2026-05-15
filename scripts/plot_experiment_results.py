"""
Plot key metrics summarized in EXPERIMENT_PLAN_RESULTS.md and the new 4.4/4.5
sections of EXPERIMENT_RESULTS_SUMMARY.md.

This script is deliberately simple: numbers are hard-coded from the experiment summary
so plots stay stable and fast to generate without re-parsing markdown.
Outputs are written to eval/plots/*.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "eval" / "plots"


def _save(fig, name: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _annotate_bars(ax, bars, fmt="{:.2f}"):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_retrieval_full() -> Path:
    metrics = ["Recall@5", "MRR@5", "nDCG@5", "R@1", "R@3"]
    clean = [0.9533, 0.4988, 0.6114, 0.2430, 0.7103]
    poisoned = [0.5607, 0.3210, 0.3803, 0.1963, 0.4393]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.2, 4))
    bars_clean = ax.bar(x - width / 2, clean, width, label="Clean index", color="#4CAF50")
    bars_poison = ax.bar(x + width / 2, poisoned, width, label="Poisoned index", color="#F44336")

    ax.set_ylabel("Score")
    ax.set_title("Retrieval (106 samples, hybrid, k=5)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.legend()

    _annotate_bars(ax, bars_clean)
    _annotate_bars(ax, bars_poison)
    return _save(fig, "retrieval_full_clean_vs_poisoned.png")


def plot_retrieval_pilot() -> Path:
    metrics = ["Recall@5", "MRR@5", "nDCG@5", "R@1", "R@3"]
    methods = {
        "Hybrid (clean)": [1.0000, 0.5067, 0.6297, 0.20, 0.80],
        "Hybrid + rerank": [1.0000, 0.5367, 0.6510, 0.30, 0.70],
        "Hybrid (poisoned)": [0.8000, 0.4283, 0.5210, 0.20, 0.60],
    }

    x = np.arange(len(metrics))
    width = 0.22
    colors = ["#4CAF50", "#2196F3", "#F44336"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, values) in enumerate(methods.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=label, color=colors[i])
        _annotate_bars(ax, bars)

    ax.set_ylabel("Score")
    ax.set_title("Retrieval pilot (10 samples, k=5)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.legend()
    return _save(fig, "retrieval_pilot_methods.png")


def plot_query_rewrite() -> Path:
    labels = ["Original", "Condense", "HyDE"]
    recall = [1.0, 1.0, 0.6]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(x, recall, color="#9C27B0")
    ax.set_ylabel("Recall@5")
    ax.set_title("Query rewrite ablation (pilot 10, k=5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    _annotate_bars(ax, bars)
    return _save(fig, "retrieval_query_rewrite.png")


def plot_generation_f1_halluc() -> Path:
    configs = [
        "RAG part1",
        "Closed part1",
        "RAG part2",
        "Closed part2",
        "RAG rewrite20",
        "Closed rewrite20",
    ]
    f1 = [0.1129, 0.0787, 0.0770, 0.0739, 0.1036, 0.0535]
    halluc = [0.1443, 1.0000, 0.1411, 1.0000, 0.1464, 1.0000]

    x = np.arange(len(configs))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    bars_f1 = axes[0].bar(x, f1, width, color="#2196F3")
    axes[0].set_title("F1 by setting (k=2 unless closed)")
    axes[0].set_ylabel("F1")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs, rotation=25, ha="right")
    axes[0].set_ylim(0, 0.3)
    _annotate_bars(axes[0], bars_f1)

    bars_hal = axes[1].bar(x, halluc, width, color="#FF9800")
    axes[1].set_title("Hallucination rate")
    axes[1].set_ylabel("Rate")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(configs, rotation=25, ha="right")
    axes[1].set_ylim(0, 1.1)
    _annotate_bars(axes[1], bars_hal, fmt="{:.2f}")

    fig.tight_layout()
    return _save(fig, "generation_f1_hallucination.png")


def plot_prompt_compare() -> Path:
    metrics = ["F1", "Hallucination"]
    strict = [0.0937, 0.1176]
    vanilla = [0.0948, 0.1758]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    bars_strict = ax.bar(x - width / 2, strict, width, label="Strict prompt", color="#607D8B")
    bars_vanilla = ax.bar(x + width / 2, vanilla, width, label="Vanilla prompt", color="#FFC107")

    ax.set_title("Prompt comparison (full 106, RAG, k=2)")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.3)
    ax.legend()

    _annotate_bars(ax, bars_strict)
    _annotate_bars(ax, bars_vanilla)
    return _save(fig, "generation_prompt_compare.png")


def plot_judge_scores() -> Path:
    configs = [
        "RAG full (50)",
        "Closed full (50)",
        "RAG part1 (30)",
        "Closed part1 (29)",
        "RAG part2 (30)",
        "Closed part2 (30)",
    ]
    scores = [2.90, 1.60, 3.20, 1.47, 2.18, 1.47]
    # Asymmetric CI only for full-sample runs
    lower_err = [0.64, 0.50, 0.0, 0.0, 0.0, 0.0]
    upper_err = [0.54, 0.59, 0.0, 0.0, 0.0, 0.0]

    x = np.arange(len(configs))
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(
        x,
        scores,
        yerr=[lower_err, upper_err],
        capsize=6,
        color="#3F51B5",
        alpha=0.85,
    )
    ax.set_ylabel("Judge score (1-5)")
    ax.set_title("LLM-as-Judge scores")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylim(0, 4.5)
    _annotate_bars(ax, bars, fmt="{:.2f}")
    return _save(fig, "judge_scores.png")


def plot_latency() -> Path:
    labels = ["Hybrid retrieval", "Hybrid + rerank"]
    times = [0.33, 41.66]  # seconds for 10 queries

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x, times, color=["#009688", "#E91E63"])
    ax.set_ylabel("Latency for 10 queries (s)")
    ax.set_title("Latency impact of reranking")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    _annotate_bars(ax, bars, fmt="{:.2f}")
    return _save(fig, "latency_rerank.png")


# --- New plots for EXPERIMENT_RESULTS_SUMMARY 4.4 / 4.5 ---


def plot_poison_retrieval() -> Path:
    metrics = ["Recall@5", "MRR@5", "nDCG@5", "R@1", "R@3"]
    clean = [1.0, 0.5067, 0.6297, 0.20, 0.80]
    poisoned = [0.8, 0.4283, 0.5210, 0.20, 0.60]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.2, 4))
    bars_clean = ax.bar(x - width / 2, clean, width, label="Clean", color="#4CAF50")
    bars_poison = ax.bar(x + width / 2, poisoned, width, label="Poisoned", color="#F44336")

    ax.set_ylabel("Score")
    ax.set_title("Poisoning impact on retrieval (pilot 10, k=5)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.legend()

    _annotate_bars(ax, bars_clean)
    _annotate_bars(ax, bars_poison)
    return _save(fig, "poison_retrieval.png")


def plot_poison_generation() -> Path:
    metrics = ["F1", "Hallucination"]
    clean = [0.0937, 0.1176]
    poisoned = [0.0850, 0.1446]

    judge_configs = ["Clean", "Poisoned"]
    judge_scores = [2.90, 2.64]
    judge_lower = [0.64, 0.61]  # 2.90-2.26, 2.64-2.03
    judge_upper = [0.54, 0.62]  # 3.44-2.90, 3.26-2.64

    x = np.arange(len(metrics))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    bars_clean = axes[0].bar(x - width / 2, clean, width, label="Clean", color="#4CAF50")
    bars_poison = axes[0].bar(x + width / 2, poisoned, width, label="Poisoned", color="#F44336")
    axes[0].set_title("Generation quality (RAG, k=2)")
    axes[0].set_ylabel("Score")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylim(0, 0.25)
    axes[0].legend()
    _annotate_bars(axes[0], bars_clean)
    _annotate_bars(axes[0], bars_poison)

    jx = np.arange(len(judge_configs))
    bars_j = axes[1].bar(
        jx,
        judge_scores,
        yerr=[judge_lower, judge_upper],
        capsize=6,
        color=["#4CAF50", "#F44336"],
        alpha=0.9,
    )
    axes[1].set_title("Judge score (30-50 samples)")
    axes[1].set_ylabel("Score (1-5)")
    axes[1].set_xticks(jx)
    axes[1].set_xticklabels(judge_configs)
    axes[1].set_ylim(0, 4.2)
    _annotate_bars(axes[1], bars_j, fmt="{:.2f}")

    fig.tight_layout()
    return _save(fig, "poison_generation.png")


def plot_poison_citation_drift() -> Path:
    # Percentage of retrieved docs that are poisoned
    value_pct = 6.5

    fig, ax = plt.subplots(figsize=(4.2, 4))
    bars = ax.bar([0], [value_pct], color="#9C27B0")
    ax.set_title("Citation drift due to poisoning")
    ax.set_ylabel("Poisoned in retrieved set (%)")
    ax.set_xticks([0])
    ax.set_xticklabels(["Hybrid k=5"])
    ax.set_ylim(0, 10)
    _annotate_bars(ax, bars, fmt="{:.1f}%")
    return _save(fig, "poison_citation_drift.png")


def plot_pii_retrieval() -> Path:
    metrics = ["Recall@5", "MRR@5", "nDCG@5"]
    clean = [0.9057, 0.5031, 0.6036]
    redacted = [0.8302, 0.4513, 0.5458]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars_clean = ax.bar(x - width / 2, clean, width, label="Clean", color="#2196F3")
    bars_red = ax.bar(x + width / 2, redacted, width, label="Redacted", color="#FF5722")

    ax.set_ylabel("Score")
    ax.set_title("PII redaction impact on retrieval (part1, k=5)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.legend()

    _annotate_bars(ax, bars_clean)
    _annotate_bars(ax, bars_red)
    return _save(fig, "pii_retrieval.png")


def plot_pii_generation() -> Path:
    metrics = ["F1", "Hallucination"]
    clean = [0.1129, 0.1443]
    redacted = [0.1012, 0.1813]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.2, 4))
    bars_clean = ax.bar(x - width / 2, clean, width, label="Clean", color="#2196F3")
    bars_red = ax.bar(x + width / 2, redacted, width, label="Redacted", color="#FF5722")

    ax.set_ylabel("Score")
    ax.set_title("PII redaction impact on generation (part1, k=2)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.25)
    ax.legend()

    _annotate_bars(ax, bars_clean)
    _annotate_bars(ax, bars_red)
    return _save(fig, "pii_generation.png")


def plot_pii_counts() -> Path:
    categories = ["Email", "Phone"]
    before = [360, 399]
    after = [0, 0]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars_before = ax.bar(x - width / 2, before, width, label="Before redact", color="#607D8B")
    bars_after = ax.bar(x + width / 2, after, width, label="After redact", color="#8BC34A")

    ax.set_ylabel("PII mentions (count)")
    ax.set_title("PII removal effectiveness (dev part1)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, max(before) * 1.1)
    ax.legend()

    _annotate_bars(ax, bars_before, fmt="{:.0f}")
    _annotate_bars(ax, bars_after, fmt="{:.0f}")
    return _save(fig, "pii_counts.png")


def main():
    outputs = [
        plot_retrieval_full(),
        plot_retrieval_pilot(),
        plot_query_rewrite(),
        plot_generation_f1_halluc(),
        plot_prompt_compare(),
        plot_judge_scores(),
        plot_latency(),
        plot_poison_retrieval(),
        plot_poison_generation(),
        plot_poison_citation_drift(),
        plot_pii_retrieval(),
        plot_pii_generation(),
        plot_pii_counts(),
    ]
    for path in outputs:
        print(f"Saved: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
