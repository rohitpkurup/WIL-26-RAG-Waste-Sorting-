# Test-Driven RAG Waste Sorting Assistant

**WIL Group:** Group 26

## Team Members

| Student ID | Name |
|---|---|
| s4192927 | Rohit  | 	GitHub and Trello administration; repository setup and version control |
| sXXXXXXX | Nihant |   Domain and data curation; knowledge base sourcing; user story design.  |
| s4090991 | Shashank | RAG pipeline development; bug fixes; evaluation experiments (chunk size / Top-K comparison) |
| sXXXXXXX | Surbhi | 	Report drafting; project aim statement; sprint planning |
| s4129561 | Shobhen | 	Report compilation and finalisation; code corrections and bug fixes; completing outstanding project aspects; Trello board setup and maintenance|

Starter implementation for a WIL project exploring **Test-Driven Retrieval-Augmented Generation (RAG)** for household waste-sorting questions.

## Project idea

The prototype retrieves relevant information from a small collection of official waste and recycling documents, then asks a local LLM to answer using only that retrieved context.

## Project Aim Statement

This project develops and evaluates a Test-Driven RAG assistant that helps Victorian households sort waste correctly using verified government sources, demonstrating measurable value through a rigorous evaluation framework rather than simply building a chatbot.

## User Stories

- As a household resident, I want to ask plain-language questions about how to dispose of an item, so that I can sort my waste correctly without searching through council PDFs.

- As a household resident, I want to receive a clear "I don't know" response when my question falls outside official waste-sorting guidance, so that I do not act on a guessed or incorrect answer.

- As a local council communications officer, I want to see which official source document supports each answer, so that I can trust and verify the information given to residents.

- As a sustainability education coordinator, I want to deploy a lightweight Q&A assistant trained only on verified government sources, so that I can reduce contamination in recycling and FOGO streams without hiring extra support staff.

- As a project evaluator (marker or mentor), I want to see quantitative evidence of retrieval accuracy and refusal behaviour, so that I can assess whether the RAG solution demonstrably adds value rather than simply resembling a chatbot.

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
│   ├── evaluate.py
│   └── experiment.py
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

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
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

For Milestone 1, around 3-6 documents is sufficient. Four Sustainability Victoria sources are currently included, covering recycling bins, FOGO, e-waste, and special or hazardous waste.

## 4. Ingest and chunk documents

```bash
python src/ingest.py
```

This produces:

```text
data/processed/chunks.json
```

Current default settings, updated following the comparison experiment described below:
- chunk size = 350 characters
- overlap = 120 characters
- Top-K = 5

These were originally set to chunk size 700 and Top-K 3 as starter values. After running a nine-configuration comparison experiment (see `src/experiment.py` and `results/experiment_comparison.csv`), the team adopted chunk size 350 and Top-K 5, which produced the highest retrieval confidence and category keyword match while maintaining 100 percent refusal accuracy on out-of-scope questions.

These values remain exposed as variables in `src/ingest.py` and `src/rag.py` so the team can test further configurations later.

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

## 7. Run the baseline evaluation

```bash
python src/evaluate.py
```

Results are written to:

```text
results/baseline_results.csv
```

The current automated `category_keyword_match` measure is intentionally basic and should not be treated as the final project correctness measure.

A more robust team-designed evaluation could incorporate:
- manually created reference answers
- expected source or document labels
- human correctness judgements
- retrieval Recall@K / Precision@K
- faithfulness checks
- source attribution correctness
- unsupported-question refusal rate

## Chunk size and Top-K comparison experiment

This experiment has been completed. `src/experiment.py` automatically tests all combinations of the following parameters:
- Chunk sizes: 350, 700, 1000
- Top-K values: 1, 3, 5

across the full 15-question test set, producing `results/experiment_comparison.csv`.

Result: chunk size 350 combined with Top-K 5 produced the best combination of retrieval confidence (average score 0.45) and category keyword match (60 percent), while maintaining 100 percent refusal accuracy. This configuration is now the project default, as noted above.

Known limitation: because Ollama generation is non-deterministic, repeated evaluation runs on the same 15 questions showed refusal accuracy varying between 93.3 and 100 percent, and keyword match varying between 53.3 and 60 percent. The figures reported above represent the majority result across three repeated runs.

To re-run the full comparison:

```bash
python src/experiment.py
```

Note that this overwrites `chunks.json` and `retrieval_index.pkl` with the last configuration tested. Run `python src/ingest.py` and `python src/retrieve.py` afterward to restore the project defaults (chunk size 350) for normal use.

## Handover status

### Already prepared
- starter project structure
- document ingestion
- chunking, including word-boundary-safe chunk splitting
- semantic retrieval
- local Ollama generation
- unsupported-answer guard
- deterministic, de-duplicated source attribution
- sample test collection
- baseline evaluation script
- chunk size and Top-K comparison experiment, with results recorded

### Team next steps

1. Agree on the final domain scope and stakeholder. Complete: Victorian household waste sorting.
2. Add 3-6 authoritative waste-sorting documents. Complete: four Sustainability Victoria sources added.
3. Record document URLs and versions in a source log. Outstanding.
4. Run the baseline pipeline. Complete.
5. Check retrieval outputs manually. Complete, during experiment review.
6. Add expected source labels to the test collection. Outstanding; would enable proper Recall@K and Precision@K measurement.
7. Improve the evaluation framework beyond keyword matching. Outstanding; planned for the next sprint.
8. Compare at least two RAG configurations. Complete: nine configurations tested, see experiment results.
9. Record results in GitHub and Trello. In progress.
10. Decide the final design and assessable conclusions as a team. Ongoing.

## 3-Week Plan

### Week 1: Strengthen the evaluation framework and knowledge base
- Add expected source/document labels to `test_questions.csv` to enable proper Recall@K and Precision@K measurement, not just refusal accuracy.
- Design and implement a stronger correctness measure to replace `category_keyword_match` (e.g. manually written reference answers with human correctness scoring).
- Record source URLs and last-updated dates for all knowledge base documents in a source log, to support source-attribution evaluation.
- Expand the test question set beyond 15 questions to cover more edge cases (ambiguous items, multi-category items, regional variation).

### Week 2: Expand evaluation dimensions and stress-test the pipeline
- Implement faithfulness checks (does the answer actually reflect the retrieved context, or does it hallucinate beyond it).
- Add source attribution correctness scoring (does the cited source actually support the claim made).
- Test the pipeline against a larger, more varied knowledge base (adding 2-3 more official documents) to check retrieval still performs well at scale.
- Begin drafting the technical report / presentation materials summarising evaluation methodology and results so far.

### Week 3: Finalise design, consolidate findings, and prepare deliverables
- Compare final evaluation results across all implemented dimensions and settle on the definitive configuration and evaluation framework as a team.
- Conduct a final round of manual review across all test questions to sanity-check automated evaluation scores.
- Consolidate all findings (chunk-size/Top-K comparison, faithfulness, source attribution, refusal accuracy) into a single results summary.
- Finalise documentation (README, Trello board, GitHub commit history) and prepare the demo/prototype walkthrough for presentation.


## Bottlenecks and Risks

- **Non-deterministic LLM output.** Ollama's local generation is not seeded, so repeated evaluation runs on the same question set produce slightly different results (refusal accuracy varied 93.3-100 percent, keyword match varied 53.3-60 percent across three runs). This makes single-run results unreliable and requires repeated-run averaging or a fixed seed going forward.

- **Weak automated correctness measure.** The current `category_keyword_match` proxy checks for literal keyword presence and does not reliably reflect true answer correctness. Several genuinely correct answers were marked as failures purely because they used different wording to the expected category label. This risks understating the system's real performance if not replaced with a stronger measure.

- **Small knowledge base.** With only four source documents and 13 chunks (at the current chunk size), retrieval has limited material to work with. This keeps the prototype manageable for Milestone 1, but retrieval performance may not generalise to a larger, more realistic knowledge base.

- **Limited test question coverage.** The 15-question test set does not yet cover ambiguous or multi-category waste items (e.g. items that are partly recyclable and partly hazardous), which are common in real-world use and would test the system's limits more thoroughly.

- **Reliance on individual team members' local setup.** Since the project depends on a local Ollama installation and model, results may vary slightly between team members' machines. This is mitigated by documenting exact setup steps in the README, but is a risk for reproducibility.

- **Time constraint for evaluation-framework improvements.** Strengthening the evaluation framework (faithfulness checks, source-attribution scoring, Recall@K/Precision@K) is more time-intensive than initial pipeline development. This has been scoped into the 3-week plan, but remains a risk if the team's availability is constrained by other coursework.

  
## Important assessment note

This starter implementation is intended as code assistance for reproducing and understanding a RAG workflow. The team should review, test, modify, and document all code, and make its own final project decisions, analysis, conclusions, and assessable writing.
