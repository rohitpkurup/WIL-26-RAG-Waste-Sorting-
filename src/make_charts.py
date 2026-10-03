"""
make_charts.py

Builds the presentation charts from the evaluation results.

Usage (from the project folder, after evaluate_full.py):
    python src/make_charts.py

Outputs (in results/charts/):
    chart_system_vs_baseline.png
    chart_by_k.png
    chart_retrieval_by_k.png
and prints the headline numbers.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=16, fontweight="bold", pad=16)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])


def headline_scores():
    """Correctness of the main RAG configuration and the baseline.
    Uses human marks from review_sheet.csv when they are filled in,
    otherwise the automatic check."""
    sheet = pd.read_csv(RESULTS / "review_sheet.csv")
    human_done = sheet["human_rag_correct"].notna().all() and \
        sheet["human_baseline_correct"].notna().all()
    rag_col = "human_rag_correct" if human_done else "rag_correct_auto"
    base_col = "human_baseline_correct" if human_done else "baseline_correct_auto"

    answerable = sheet[sheet["expected_behaviour"] == "answer"]
    refuse = sheet[sheet["expected_behaviour"] == "refuse"]
    scores = {
        "Correct answers\n(%d answerable questions)" % len(answerable):
            (100 * answerable[rag_col].mean(), 100 * answerable[base_col].mean()),
        "Declines when it should\n(%d unanswerable questions)" % len(refuse):
            (100 * refuse[rag_col].mean(), 100 * refuse[base_col].mean()),
        "Overall\n(all %d questions)" % len(sheet):
            (100 * sheet[rag_col].mean(), 100 * sheet[base_col].mean()),
    }
    return scores, human_done


def chart_system_vs_baseline():
    scores, human_done = headline_scores()
    labels = list(scores)
    rag = [scores[k][0] for k in labels]
    base = [scores[k][1] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    width = 0.3
    xs = range(len(labels))
    bars_rag = ax.bar([x - width / 2 - 0.01 for x in xs], rag, width, color=BLUE,
                      label="Our RAG assistant")
    bars_base = ax.bar([x + width / 2 + 0.01 for x in xs], base, width, color=ORANGE,
                       label="Same AI model, no documents")
    for bars in (bars_rag, bars_base):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{bar.get_height():.0f}%", ha="center", color=INK, fontsize=13,
                    fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    style(ax, "Grounding the AI in official documents makes it more reliable")
    ax.set_ylim(0, 118)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False,
              labelcolor=INK_2)
    marking = "human-marked" if human_done else "automatic check (not yet human-marked)"
    fig.text(0.01, 0.01, f"Scoring: {marking}", color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(CHARTS / "chart_system_vs_baseline.png", dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    print("\nHeadline (%s)" % marking)
    for label in labels:
        print(f"  {label.splitlines()[0]:<26} RAG {scores[label][0]:.1f}%   "
              f"baseline {scores[label][1]:.1f}%")


def line_chart(table, x, series, title, xlabel, filename):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for column, label, colour in series:
        ax.plot(table[x], table[column], color=colour, linewidth=2, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        ax.annotate(f"{table[column].iloc[-1]:.0f}%",
                    (table[x].iloc[-1], table[column].iloc[-1]),
                    xytext=(10, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=12)
    ax.set_xticks(list(table[x]))
    ax.set_xlabel(xlabel)
    style(ax, title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(series),
              frameon=False, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(CHARTS / filename, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)

    retrieval = pd.read_csv(RESULTS / "retrieval_by_k.csv")
    line_chart(
        retrieval, "top_k",
        [("recall_at_k_pct", "Right document found (Recall@K)", BLUE),
         ("precision_at_k_pct", "Relevant passages (Precision@K)", ORANGE)],
        "More passages find the right document, but add noise",
        "Number of passages retrieved (Top-K)", "chart_retrieval_by_k.png")
    print("Saved chart_retrieval_by_k.png")

    if not (RESULTS / "summary.csv").exists():
        print("summary.csv not found: run the full evaluate_full.py for the other charts.")
        return

    summary = pd.read_csv(RESULTS / "summary.csv")
    by_k = summary[summary["top_k"] > 0].sort_values("top_k")
    line_chart(
        by_k, "top_k",
        [("answer_correct_auto_pct", "Correct answers", BLUE),
         ("source_precision_pct", "Relevant cited sources", ORANGE),
         ("not_in_kb_refused_pct", "Declines unsupported waste questions", AQUA)],
        "Choosing Top-K: answer quality against source quality",
        "Number of passages retrieved (Top-K)", "chart_by_k.png")
    print("Saved chart_by_k.png")

    chart_system_vs_baseline()
    print("Saved chart_system_vs_baseline.png")
    print(f"\nCharts are in: {CHARTS}")


if __name__ == "__main__":
    main()
