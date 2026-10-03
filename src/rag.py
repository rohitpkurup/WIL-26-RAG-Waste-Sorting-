import os
import requests
from retrieve import Retriever

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

# Fixed temperature and seed make repeated runs give the same answers.
# This removes the run-to-run variation recorded in Milestone 1.
OLLAMA_OPTIONS = {"temperature": 0, "seed": 42}

DEFAULT_TOP_K = 8
MIN_RETRIEVAL_SCORE = 0.10

REFUSAL_TEXT = "I cannot answer this from the available waste-sorting knowledge base."

SYSTEM_INSTRUCTION = f"""
You are a waste-sorting information assistant for households in the state of
Victoria, Australia, in the present day.

Answer ONLY from the supplied context. Do not use outside knowledge.
If the context does not mention the item or topic in the question,
say exactly:

{REFUSAL_TEXT}

Do not guess.
Answer in one or two complete sentences that say where the item goes or what
to do, using the facts in the context.
Only if the question is a yes/no question, start with "Yes" or "No" and then
give the reason from the context. Never answer with only "Yes" or "No".
Do not mention the context, context numbers or similarity scores.
Do not include a Sources line yourself; it will be added automatically.
""".strip()

# Used only for the no-retrieval baseline: the same model with no documents.
BASELINE_INSTRUCTION = """
You are a waste-sorting information assistant for households in the state of
Victoria, Australia, in the present day.

Answer the question in one or two complete sentences that say where the item
goes or what to do.
Only if the question is a yes/no question, start with "Yes" or "No" and then
give the reason. Never answer with only "Yes" or "No".
""".strip()


def build_context(results):
    parts = []
    for item in results:
        parts.append(f"[Source: {item['source']}]\n{item['text']}")
    return "\n\n".join(parts)


def generate_with_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": OLLAMA_OPTIONS,
        },
        timeout=300
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def is_refusal(text: str) -> bool:
    return "cannot answer this from the available" in text.lower()


def unique_sources(results):
    seen = []
    for item in results:
        if item["source"] not in seen:
            seen.append(item["source"])
    return seen


def answer_without_retrieval(question: str):
    """Baseline: ask the same model the same question with no documents."""
    prompt = f"{BASELINE_INSTRUCTION}\n\nQUESTION\n--------\n{question}\n\nANSWER\n------"
    raw_answer = generate_with_ollama(prompt)
    return {
        "question": question,
        "answer": raw_answer,
        "answer_body": raw_answer,
        "retrieved": [],
        "sources": [],
        "refused": is_refusal(raw_answer),
    }


class WasteSortingRAG:
    def __init__(self):
        self.retriever = Retriever()

    def answer(self, question: str, top_k: int = DEFAULT_TOP_K):
        results = self.retriever.search(question, top_k=top_k)

        if not results or results[0]["score"] < MIN_RETRIEVAL_SCORE:
            return {
                "question": question,
                "answer": REFUSAL_TEXT,
                "answer_body": REFUSAL_TEXT,
                "retrieved": results,
                "sources": [],
                "refused": True
            }

        context = build_context(results)

        prompt = f"""
{SYSTEM_INSTRUCTION}

CONTEXT
-------
{context}

QUESTION
--------
{question}

ANSWER
------
""".strip()

        raw_answer = generate_with_ollama(prompt)
        refused = is_refusal(raw_answer)

        # Strip any Sources line the model wrote anyway.
        answer_body = raw_answer.split("Sources:")[0].strip()

        if refused:
            answer_body = REFUSAL_TEXT
            answer = REFUSAL_TEXT
            sources = []
        else:
            # Sources come from what was actually retrieved, not what the model claims.
            sources = unique_sources(results)
            sources_line = "Sources:\n" + "\n".join(f"- {s}" for s in sources)
            answer = f"{answer_body}\n\n{sources_line}"

        return {
            "question": question,
            "answer": answer,
            "answer_body": answer_body,
            "retrieved": results,
            "sources": sources,
            "refused": refused
        }


def main():
    rag = WasteSortingRAG()

    print("Waste Sorting RAG")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        result = rag.answer(question)

        print("\nAnswer:")
        print(result["answer"])

        print("\nRetrieved:")
        for item in result["retrieved"]:
            print(
                f"- {item['source']} | "
                f"chunk={item['chunk_index']} | "
                f"score={item['score']:.4f}"
            )

        print()


if __name__ == "__main__":
    main()
