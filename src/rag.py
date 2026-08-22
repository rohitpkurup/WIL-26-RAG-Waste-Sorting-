import requests
from retrieve import Retriever

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

MIN_RETRIEVAL_SCORE = 0.10

SYSTEM_INSTRUCTION = """
You are a waste-sorting information assistant.

Answer ONLY from the supplied context.
If the context does not contain enough information to answer the question,
say exactly:

I cannot answer this from the available waste-sorting knowledge base.

Do not guess.
Keep the answer concise.
Do not include a Sources line yourself — it will be added automatically.
""".strip()


def build_context(results):
    parts = []
    for i, item in enumerate(results, start=1):
        parts.append(
            f"[Context {i}]\n"
            f"Source: {item['source']}\n"
            f"Similarity: {item['score']:.4f}\n"
            f"{item['text']}"
        )
    return "\n\n".join(parts)


def generate_with_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"].strip()


class WasteSortingRAG:
    def __init__(self):
        self.retriever = Retriever()

    def answer(self, question: str, top_k: int = 3):
        results = self.retriever.search(question, top_k=top_k)

        if not results or results[0]["score"] < MIN_RETRIEVAL_SCORE:
            return {
                "question": question,
                "answer": "I cannot answer this from the available waste-sorting knowledge base.",
                "retrieved": results,
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

        refused = raw_answer.startswith(
            "I cannot answer this from the available waste-sorting knowledge base."
        )

        # Strip any Sources line the model wrote anyway — unreliable/can duplicate.
        answer_body = raw_answer.split("Sources:")[0].strip()

        if refused:
            answer = answer_body
        else:
            # Build a clean, de-duplicated, ordered source list from what was
            # actually retrieved (not what the model claims).
            seen = []
            for item in results:
                if item["source"] not in seen:
                    seen.append(item["source"])
            sources_line = "Sources:\n" + "\n".join(f"- {s}" for s in seen)
            answer = f"{answer_body}\n\n{sources_line}"

        return {
            "question": question,
            "answer": answer,
            "retrieved": results,
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
