from pathlib import Path
import json
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "data" / "knowledge_base"
OUT_FILE = ROOT / "data" / "processed" / "chunks.json"

CHUNK_SIZE = 350
CHUNK_OVERLAP = 120


def read_document(path: Path) -> str:
    """Read a supported PDF or TXT document."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n".join(pages)

    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    return ""


def clean_text(text: str) -> str:
    return " ".join(text.split())


def make_chunks(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple overlapping character-based chunking, snapped to word boundaries."""
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunks.append(text[start:end].strip())

        if end >= len(text):
            break

        next_start = max(0, end - overlap)

        if next_start > start:
            space_after = text.find(" ", next_start)
            if space_after != -1 and space_after < end:
                next_start = space_after + 1

        start = next_start

    return chunks


def main():
    documents = [
        p for p in KB_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}
        and p.name != "ADD_DOCUMENTS_HERE.txt"
    ]

    if not documents:
        print(f"No PDF/TXT knowledge-base documents found in: {KB_DIR}")
        print("Add official waste-sorting documents, then run this script again.")
        return

    records = []

    for path in documents:
        text = read_document(path)
        chunks = make_chunks(text)

        for i, chunk in enumerate(chunks):
            records.append({
                "chunk_id": f"{path.stem}_{i:04d}",
                "source": path.name,
                "chunk_index": i,
                "text": chunk
            })

        print(f"{path.name}: {len(chunks)} chunks")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nSaved {len(records)} chunks to {OUT_FILE}")


if __name__ == "__main__":
    main()
