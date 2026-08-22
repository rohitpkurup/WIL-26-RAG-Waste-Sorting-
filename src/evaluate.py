from pathlib import Path
import pandas as pd
from rag import WasteSortingRAG

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "data" / "test_questions.csv"
RESULT_FILE = ROOT / "results" / "baseline_results.csv"


def contains_expected_category(answer: str, expected: str) -> bool:
    """
    Very simple starter proxy for correctness.
    The team should replace this with a stronger human-labelled or
    reference-answer evaluation later.
    """
    if expected.lower() == "unsupported":
        return True

    return expected.lower() in answer.lower()


def main():
    tests = pd.read_csv(TEST_FILE)
    rag = WasteSortingRAG()

    output = []

    for _, row in tests.iterrows():
        result = rag.answer(row["question"], top_k=5)

        expected_refusal = row["expected_behaviour"].lower() == "refuse"
        actual_refusal = bool(result["refused"])

        refusal_correct = expected_refusal == actual_refusal

        category_match = (
            True if expected_refusal
            else contains_expected_category(
                result["answer"],
                row["expected_category"]
            )
        )

        top_source = (
            result["retrieved"][0]["source"]
            if result["retrieved"] else ""
        )

        top_score = (
            result["retrieved"][0]["score"]
            if result["retrieved"] else None
        )

        output.append({
            "id": row["id"],
            "question": row["question"],
            "expected_category": row["expected_category"],
            "expected_behaviour": row["expected_behaviour"],
            "answer": result["answer"],
            "actual_refusal": actual_refusal,
            "refusal_correct": refusal_correct,
            "category_keyword_match": category_match,
            "top_retrieved_source": top_source,
            "top_retrieval_score": top_score,
        })

        print(f"Completed {row['id']}")

    results = pd.DataFrame(output)
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_FILE, index=False)

    refusal_accuracy = results["refusal_correct"].mean() * 100
    rough_answer_match = results["category_keyword_match"].mean() * 100

    print("\nBaseline evaluation")
    print("-------------------")
    print(f"Refusal behaviour accuracy: {refusal_accuracy:.1f}%")
    print(f"Starter category keyword match: {rough_answer_match:.1f}%")
    print(f"\nFull results saved to: {RESULT_FILE}")
    print("\nNOTE: category_keyword_match is only a temporary proxy.")
    print("The team should replace it with reference answers or human judgement.")


if __name__ == "__main__":
    main()
