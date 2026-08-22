"""
experiment.py

Runs the RAG pipeline across multiple chunk-size / Top-K configurations
and produces a single comparison CSV, as suggested in the README's
"Suggested first experiment" section.

Usage:
    python src/experiment.py
"""

from pathlib import Path
import json
import pandas as pd

import ingest
from retrieve import build_index, Retriever
from rag import WasteSortingRAG
from evaluate import contains_expected_category

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "data" / "test_questions.csv"
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.json"
RESULT_FILE = ROOT / "results" / "experiment_comparison.csv"

CHUNK_SIZES = [350, 700, 1000]
TOP_K_VALUES = [1, 3, 5]
CHUNK_OVERLAP = 120


def build_chunks_for_size(chunk_size: int):
    """Re-run ingestion with a specific chunk size and save chunks.json."""
    documents = [
        p for p in ingest.KB_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}
        and p.name != "ADD_DOCUMENTS_HERE.txt"
    ]

    records = []
    for path in documents:
        text = ingest.read_document(path)
        chunks = ingest.make_chunks(text, chunk_size=chunk_size, overlap=CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            records.append({
                "chunk_id": f"{path.stem}_{i:04d}",
                "source": path.name,
                "chunk_index": i,
                "text": chunk
            })

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_FILE.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return len(records)


def run_config(chunk_size: int, top_k: int, tests: pd.DataFrame):
    """Run the full pipeline for one (chunk_size, top_k) combination."""
    print(f"\n=== chunk_size={chunk_size}, top_k={top_k} ===")

    num_chunks = build_chunks_for_size(chunk_size)
    print(f"Rebuilt {num_chunks} chunks for chunk_size={chunk_size}")

    build_index()

    rag = WasteSortingRAG()

    refusal_correct_count = 0
    category_match_count = 0
    top_scores = []

    for _, row in tests.iterrows():
        result = rag.answer(row["question"], top_k=top_k)

        expected_refusal = row["expected_behaviour"].lower() == "refuse"
        actual_refusal = bool(result["refused"])
        refusal_correct = expected_refusal == actual_refusal
        refusal_correct_count += int(refusal_correct)

        category_match = (
            True if expected_refusal
            else contains_expected_category(result["answer"], row["expected_category"])
        )
        category_match_count += int(category_match)

        if result["retrieved"]:
            top_scores.append(result["retrieved"][0]["score"])

        print(f"  {row['id']} done")

    n = len(tests)
    avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

    return {
        "chunk_size": chunk_size,
        "top_k": top_k,
        "num_chunks": num_chunks,
        "refusal_accuracy_pct": round(100 * refusal_correct_count / n, 1),
        "category_keyword_match_pct": round(100 * category_match_count / n, 1),
        "avg_top_retrieval_score": round(avg_top_score, 4),
    }


def main():
    tests = pd.read_csv(TEST_FILE)

    summary_rows = []
    for chunk_size in CHUNK_SIZES:
        for top_k in TOP_K_VALUES:
            summary_rows.append(run_config(chunk_size, top_k, tests))

    summary = pd.DataFrame(summary_rows)
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT_FILE, index=False)

    print("\n\nComparison complete.")
    print(summary.to_string(index=False))
    print(f"\nSaved comparison table to: {RESULT_FILE}")
    print(
        "\nNOTE: this experiment overwrites data/processed/chunks.json and "
        "retrieval_index.pkl with the LAST configuration tested. "
        "Run 'python src/ingest.py' and 'python src/retrieve.py' afterward "
        "to restore your default chunk_size=700 setup for normal use."
    )


if __name__ == "__main__":
    main()
