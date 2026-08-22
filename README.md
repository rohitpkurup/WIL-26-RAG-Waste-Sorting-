# Test-Driven RAG Waste Sorting Assistant

**WIL Group:** Group 26

## Team Members

| Student ID | Name |
|---|---|
| s4192927 | Rohit  |
| sXXXXXXX | Nihant |
| sXXXXXXX | Shashank |
| sXXXXXXX | Surbhi |
| s4129561 | Shobhen |

Starter implementation for a WIL project exploring **Test-Driven Retrieval-Augmented Generation (RAG)** for household waste-sorting questions.

## Project idea

The prototype retrieves relevant information from a small collection of official waste and recycling documents, then asks a local LLM to answer using only that retrieved context.

The key project focus is **evaluation**, not simply chatbot development.

## User Stories

- **As a** household resident, **I want to** ask plain-language questions about how to dispose of an item, **so that** I can sort my waste correctly without searching through council PDFs.

- **As a** household resident, **I want to** receive a clear "I don't know" when my question falls outside official waste-sorting guidance, **so that** I don't act on a guessed or incorrect answer.

- **As a** local council communications officer, **I want to** see which official source document supports each answer, **so that** I can trust and verify the information given to residents.

- **As a** sustainability education coordinator, **I want to** deploy a lightweight Q&A assistant trained only on verified government sources, **so that** I can reduce contamination in recycling/FOGO streams without hiring extra support staff.

- **As a** project evaluator (marker/mentor), **I want to** see quantitative evidence of retrieval accuracy and refusal behaviour, **so that** I can assess whether the RAG solution demonstrably adds value rather than just "looking like a chatbot."

Initial evaluation ideas:
- retrieval relevance / retrieval accuracy
- answer correctness
- correct source attribution
- unsupported-question refusal accuracy
- unanswered-question rate
- comparison of chunk size and Top-K settings

## Current starter pipeline

```text
Official documents
      |
      v
PDF/TXT ingestion
      |
      v
Text chunks
      |
      v
Sentence Transformer embeddings
      |
      v
Cosine-similarity retrieval
      |
      v
Top-K context
      |
      v
Local Ollama LLM
      |
      v
Grounded answer + source
```

## Folder structure

```text
waste-sorting-rag/
├── README.md
├── requirements.txt
├── data/
│   ├── knowledge_base/
│   ├── processed/
│   └── test_questions.csv
├── src/
│   ├── ingest.py
│   ├── retrieve.py
│   ├── rag.py
│   └── evaluate.py
└── results/
```

## 1. Set up Python

Recommended: Python 3.11 or 3.12.

Create a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, Command Prompt can be used:

```cmd
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## 2. Install Ollama

Install Ollama separately, then pull a small local model.

Example:

```bash
ollama pull llama3.2:3b
```

The starter code expects:

```text
llama3.2:3b
```

Change `OLLAMA_MODEL` in `src/rag.py` if the team chooses another model.

## 3. Add documents

Put a small set of official PDF or TXT waste-sorting documents into:

```text
data/knowledge_base/
```

For Milestone 1, around 3–6 documents is enough.

## 4. Ingest and chunk documents

```bash
python src/ingest.py
```

This produces:

```text
data/processed/chunks.json
```

Default starter settings:
- chunk size = 700 characters
- overlap = 120 characters

These are deliberately exposed in `src/ingest.py` so the team can later test different values.

## 5. Build retrieval index

```bash
python src/retrieve.py
```

This creates embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

and saves:

```text
data/processed/retrieval_index.pkl
```

## 6. Ask questions interactively

Make sure Ollama is running, then:

```bash
python src/rag.py
```

Example questions:

```text
Where should an old laptop go?
Can food scraps go in the green bin?
How do I renew my passport?
```

The last question is intentionally outside the knowledge base and should be refused.

## 7. Run the starter evaluation

```bash
python src/evaluate.py
```

Results are written to:

```text
results/baseline_results.csv
```

The current automated `category_keyword_match` is intentionally basic.
It should NOT be treated as the final project correctness measure.

A better team-designed evaluation could use:
- manually created reference answers
- expected source/document labels
- human correctness judgements
- retrieval Recall@K / Precision@K
- faithfulness checks
- source attribution correctness
- unsupported-question refusal rate

## Suggested first experiment

Run the same test questions under:

```text
Top-K = 1
Top-K = 3
Top-K = 5
```

Compare:
- retrieval relevance
- answer correctness
- refusal behaviour

Then test different chunk sizes, such as:

```text
350
700
1000
```

This gives a simple **build -> test -> change -> test again** workflow.

## Handover status

### Already prepared
- starter project structure
- document ingestion
- chunking
- semantic retrieval
- local Ollama generation
- unsupported-answer guard
- sample test collection
- baseline evaluation script

### Team next steps
1. Agree on the final domain scope and stakeholder.
2. Add 3–6 authoritative waste-sorting documents.
3. Record document URLs and versions.
4. Run the baseline pipeline.
5. Check retrieval outputs manually.
6. Add expected source labels to the test collection.
7. Improve the evaluation framework.
8. Compare at least two RAG configurations.
9. Record results in GitHub/Trello.
10. Decide the final design and assessable conclusions as a team.

## Important assessment note

This starter implementation is intended as code assistance for reproducing and understanding a RAG workflow. The team should review, test, modify, and document all code, and make its own final project decisions, analysis, conclusions, and assessable writing.
