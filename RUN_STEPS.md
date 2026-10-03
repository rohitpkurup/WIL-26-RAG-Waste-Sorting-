# Run steps: improved evaluation, charts and demo app

Run every command from the project folder, with the virtual environment active
and Ollama running.

## 1. Install the two new libraries

    pip install -r requirements.txt

## 2. Check retrieval only (about a minute, Ollama not needed)

    python src/evaluate_full.py --retrieval-only

Rebuilds the chunks and index for the 11-document knowledge base and prints
Recall@K, Precision@K and MRR for K = 1 to 15.

## 3. Run the full evaluation (roughly 20 to 40 minutes)

    python src/evaluate_full.py

Runs 48 questions through the RAG system at K = 1, 3, 5, 8, 10 and through the
same model with no documents (the baseline). Results go to `results/`.

## 4. Mark the answers by hand

Open `results/review_sheet.csv` in Excel. For each row compare the answers with
`reference_answer` and type 1 (correct) or 0 (wrong) in `human_rag_correct` and
`human_baseline_correct`. For "refuse" questions, 1 means it declined. Save as CSV.

## 5. Build the charts

    python src/make_charts.py

Charts are saved in `results/charts/`. The headline numbers are printed.

## 6. Run the demo app

    streamlit run app.py

Opens in the browser. Record the screen for the video.

## What changed since Milestone 1

- Knowledge base: 4 documents (13 chunks) to 11 documents (68 chunks). Sources and
  dates are in `data/source_log.csv`.
- Test set: 15 to 48 questions, with expected source labels, reference answers,
  tricky items and in-domain questions the knowledge base cannot answer.
  The original 15 are kept in `data/test_questions_milestone1.csv`.
- `src/rag.py`: fixed temperature and seed (repeatable results), cleaner prompt,
  and a no-retrieval baseline.
- New: `src/evaluate_full.py`, `src/make_charts.py`, `app.py`.
