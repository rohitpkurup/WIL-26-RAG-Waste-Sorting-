from pathlib import Path
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.json"
INDEX_FILE = ROOT / "data" / "processed" / "retrieval_index.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_index():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "chunks.json not found. Run: python src/ingest.py"
        )

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError("No chunks found.")

    model = SentenceTransformer(MODEL_NAME)
    texts = [item["text"] for item in chunks]

    embeddings = model.encode_document(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    payload = {
        "model_name": MODEL_NAME,
        "chunks": chunks,
        "embeddings": np.asarray(embeddings)
    }

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved retrieval index to {INDEX_FILE}")
    print(f"Indexed {len(chunks)} chunks.")


class Retriever:
    def __init__(self):
        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                "retrieval_index.pkl not found. Run: python src/retrieve.py"
            )

        with open(INDEX_FILE, "rb") as f:
            payload = pickle.load(f)

        self.chunks = payload["chunks"]
        self.embeddings = payload["embeddings"]
        self.model = SentenceTransformer(payload["model_name"])

    def search(self, query: str, top_k: int = 3):
        query_embedding = self.model.encode_query(
            [query],
            normalize_embeddings=True
        )

        scores = cosine_similarity(
            query_embedding,
            self.embeddings
        )[0]

        indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in indices:
            item = dict(self.chunks[int(idx)])
            item["score"] = float(scores[int(idx)])
            results.append(item)

        return results


def main():
    build_index()


if __name__ == "__main__":
    main()
