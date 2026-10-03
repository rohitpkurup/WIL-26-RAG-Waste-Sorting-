"""
evaluate_full.py

Full evaluation for the Test-Driven RAG waste-sorting assistant.

It compares:
  * the RAG system at several Top-K values, and
  * a no-retrieval baseline (the same LLM with no documents),
on the labelled test set in data/test_questions.csv.

Usage (from the project folder):
    python src/evaluate_full.py --retrieval-only     # seconds, no Ollama needed
    python src/evaluate_full.py                      # full run, needs Ollama

Outputs (in results/):
    retrieval_by_k.csv   Recall@K, Precision@K and MRR for K = 1..15
    full_results.csv     one row per question per configuration
    summary.csv          one row per configuration
    review_sheet.csv     answers side by side for human marking
"""

from pathlib import Path
import argparse
import re
import time

import pandas as pd

import ingest
from retrieve import build_index, Retriever
from rag import WasteSortingRAG, answer_without_retrieval, DEFAULT_TOP_K

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "data" / "test_questions.csv"
RESULTS = ROOT / "results"

TOP_K_VALUES = [1, 3, 5, 8, 10]
RETRIEVAL_K_VALUES = list(range(1, 16))

NO_WORDS = (
    r"\b(no|not|never|cannot|can't|cant|don't|shouldn't|mustn't|"
    r"keep (it|them|these|this) out)\b"
)
YES_WORDS = r"\b(yes|can go|can be|is accepted|are accepted|belongs?)\b"

STOPWORDS = set("""
a an and are as at be been but by can for from has have if in into is it its
of on or should so such that the their them then there these they this to was
were what when where which will with you your not no yes any do does go goes
put take also only more than other
""".split())


# ---------- scoring helpers ----------

def split_list(value):
    if pd.isna(value) or not str(value).strip():
        return []
    return [v.strip() for v in str(value).split("|") if v.strip()]


def matches(term: str, answer: str) -> bool:
    text = answer.lower().strip()
    if term == "@NO":
        return re.search(NO_WORDS, text) is not None
    if term == "@YES":
        return re.search(YES_WORDS, text) is not None
    if term == "@STARTS_NO":
        return re.match(r"\W*no\b", text) is not None
    if term == "@STARTS_YES":
        return re.match(r"\W*yes\b", text) is not None
    return re.search(r"\b" + re.escape(term.lower()) + r"\b", text) is not None


def auto_correct(answer: str, accept, reject) -> bool:
    """Automatic correctness check. A proxy: the headline figure should use
    the human marks entered in review_sheet.csv."""
    if any(matches(t, answer) for t in reject):
        return False
    return any(matches(t, answer) for t in accept)


def content_words(text: str):
    words = re.findall(r"[a-z][a-z\-]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def grounding(answer: str, retrieved) -> float:
    """Share of the answer's content words that appear in the retrieved text.
    A simple faithfulness proxy: low values suggest the answer goes beyond
    its sources."""
    words = content_words(answer)
    if not words:
        return float("nan")
    context = " ".join(item["text"] for item in retrieved).lower()
    return sum(w in context for w in words) / len(words)


def retrieval_metrics(retrieved, expected):
    sources = [item["source"] for item in retrieved]
    hit = any(s in expected for s in sources)
    precision = sum(s in expected for s in sources) / len(sources) if sources else 0.0
    rank = next((i + 1 for i, s in enumerate(sources) if s in expected), None)
    return hit, precision, (1.0 / rank if rank else 0.0)


# ---------- pipeline steps ----------

def rebuild_index():
    """Re-chunk the knowledge base at the default chunk size and re-embed it."""
    ingest.main()
    build_index()


def run_retrieval_only(tests):
    retriever = Retriever()
    answerable = tests[tests["expected_behaviour"] == "answer"]
    rows = []
    for k in RETRIEVAL_K_VALUES:
        hits, precisions, rrs = [], [], []
        for _, row in answerable.iterrows():
            retrieved = retriever.search(row["question"], top_k=k)
            hit, precision, rr = retrieval_metrics(retrieved, split_list(row["expected_source"]))
            hits.append(hit)
            precisions.append(precision)
            rrs.append(rr)
        rows.append({
            "top_k": k,
            "recall_at_k_pct": round(100 * sum(hits) / len(hits), 1),
            "precision_at_k_pct": round(100 * sum(precisions) / len(precisions), 1),
            "mrr": round(sum(rrs) / len(rrs), 3),
        })
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "retrieval_by_k.csv", index=False)
    print("\nRetrieval quality by Top-K "
          f"({len(answerable)} answerable questions, {len(retriever.chunks)} chunks)")
    print(table.to_string(index=False))
    return table


def score_row(row, result, config, top_k, seconds):
    expected_refusal = row["expected_behaviour"] == "refuse"
    refused = bool(result["refused"])
    expected_sources = split_list(row["expected_source"])
    body = result["answer_body"]

    out = {
        "config": config,
        "top_k": top_k,
        "id": row["id"],
        "question": row["question"],
        "question_type": row["question_type"],
        "expected_behaviour": row["expected_behaviour"],
        "reference_answer": row["reference_answer"],
        "answer": body,
        "refused": refused,
        "refusal_correct": expected_refusal == refused,
        "seconds": round(seconds, 2),
    }

    if expected_refusal:
        out["correct_auto"] = refused
    else:
        out["correct_auto"] = (not refused) and auto_correct(
            body, split_list(row["accept_any"]), split_list(row["reject_any"])
        )

    if result["retrieved"] and not expected_refusal:
        hit, precision, rr = retrieval_metrics(result["retrieved"], expected_sources)
        out["retrieval_hit"] = hit
        out["retrieval_precision"] = precision
        out["reciprocal_rank"] = rr
        if result["sources"]:
            cited = result["sources"]
            out["cited_sources"] = "|".join(cited)
            out["source_precision"] = sum(s in expected_sources for s in cited) / len(cited)
            out["grounding"] = grounding(body, result["retrieved"])
    return out


def summarise(results):
    rows = []
    for config, group in results.groupby("config", sort=False):
        answerable = group[group["expected_behaviour"] == "answer"]
        should_refuse = group[group["expected_behaviour"] == "refuse"]
        not_in_kb = group[group["question_type"] == "not_in_kb"]

        def pct(series):
            series = series.dropna()
            return round(100 * series.astype(float).mean(), 1) if len(series) else None

        def col(frame, name):
            return frame[name] if name in frame else pd.Series(dtype=float)

        rows.append({
            "config": config,
            "top_k": group["top_k"].iloc[0],
            "questions": len(group),
            "overall_correct_auto_pct": pct(group["correct_auto"]),
            "answer_correct_auto_pct": pct(answerable["correct_auto"]),
            "unanswered_pct": pct(answerable["refused"]),
            "refusal_accuracy_pct": pct(group["refusal_correct"]),
            "correct_refusals_pct": pct(should_refuse["refused"]),
            "not_in_kb_refused_pct": pct(not_in_kb["refused"]),
            "recall_at_k_pct": pct(col(answerable, "retrieval_hit")),
            "source_precision_pct": pct(col(answerable, "source_precision")),
            "grounding_pct": pct(col(answerable, "grounding")),
            "avg_seconds": round(group["seconds"].mean(), 2),
        })
    return pd.DataFrame(rows)


def write_review_sheet(results):
    """Side-by-side answers for human marking. Fill the human_* columns with
    1 (correct) or 0 (wrong), then run make_charts.py."""
    main_config = f"RAG k={DEFAULT_TOP_K}"
    rag = results[results["config"] == main_config].set_index("id")
    base = results[results["config"] == "No retrieval"].set_index("id")
    sheet = pd.DataFrame({
        "question": rag["question"],
        "question_type": rag["question_type"],
        "expected_behaviour": rag["expected_behaviour"],
        "reference_answer": rag["reference_answer"],
        "rag_answer": rag["answer"],
        "rag_correct_auto": rag["correct_auto"].astype(int),
        "human_rag_correct": "",
        "baseline_answer": base["answer"],
        "baseline_correct_auto": base["correct_auto"].astype(int),
        "human_baseline_correct": "",
    })
    path = RESULTS / "review_sheet.csv"
    if path.exists():
        # Keep marks already entered, as long as the answers have not changed.
        old = pd.read_csv(path, index_col="id")
        for side in ["rag", "baseline"]:
            for qid in sheet.index.intersection(old.index):
                if str(old.at[qid, f"{side}_answer"]) == str(sheet.at[qid, f"{side}_answer"]):
                    value = old.at[qid, f"human_{side}_correct"]
                    if not pd.isna(value):
                        sheet.at[qid, f"human_{side}_correct"] = int(value)
    sheet.to_csv(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true",
                        help="only measure retrieval (no Ollama needed)")
    parser.add_argument("--skip-rebuild", action="store_true",
                        help="reuse the existing chunks and index")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    tests = pd.read_csv(TEST_FILE)

    if not args.skip_rebuild:
        rebuild_index()

    run_retrieval_only(tests)
    if args.retrieval_only:
        return

    rag = WasteSortingRAG()
    output = []

    configs = [(f"RAG k={k}", k) for k in TOP_K_VALUES] + [("No retrieval", 0)]
    for config, k in configs:
        print(f"\n=== {config} ===")
        for _, row in tests.iterrows():
            start = time.time()
            if k == 0:
                result = answer_without_retrieval(row["question"])
            else:
                result = rag.answer(row["question"], top_k=k)
            output.append(score_row(row, result, config, k, time.time() - start))
            print(f"  {row['id']} done")
        # Save after every configuration so a crash loses little work.
        pd.DataFrame(output).to_csv(RESULTS / "full_results.csv", index=False)

    results = pd.DataFrame(output)
    summary = summarise(results)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    review_path = write_review_sheet(results)

    print("\n\nSummary")
    print("-------")
    print(summary.to_string(index=False))
    print(f"\nSaved: {RESULTS / 'full_results.csv'}")
    print(f"Saved: {RESULTS / 'summary.csv'}")
    print(f"Saved: {review_path}")
    print("\nNEXT: open review_sheet.csv, mark human_rag_correct and "
          "human_baseline_correct as 1 or 0, then run: python src/make_charts.py")


if __name__ == "__main__":
    main()
