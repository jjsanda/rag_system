# Evaluation

A small, deterministic offline harness that scores the RAG pipeline over the sample corpus,
covering both retrieval quality and answer quality.

```bash
make eval            # or: uv run python -m evaluation.run_eval --top-k 5
```

It ingests `data/corpus/` with the offline stack (hashing embedder + FAISS + extractive
answerer), runs every question in [`qa_dataset.json`](qa_dataset.json), prints a per-question
table, and writes [`results.json`](results.json) / [`results.md`](results.md).

## Metrics (all computed in pure Python; see [`run_eval.py`](run_eval.py))

| Metric | Definition |
| --- | --- |
| **hit@k** | Fraction of questions whose gold document appears in the top-k retrieved. |
| **MRR** | Mean reciprocal rank of the first gold document (`1/rank`, `0` if absent). |
| **key-fact coverage** | Fraction of each question's labelled key facts present in the answer text. |
| **groundedness** | Per answer sentence, the best lexical recall against any retrieved chunk (share of the sentence's words found in the context), averaged. Close to 1.0 for an extractive answerer; lower when an answer paraphrases or invents. |

## Dataset

`qa_dataset.json` holds 12 questions across the 8 corpus documents. Each carries
`relevant_doc_ids` (gold source documents) and `key_facts` (case-insensitive substrings a
correct answer should contain).

## Limitations

- The default embedder is a lexical hashing embedder (zero-dependency, offline). It has
  no semantics or stemming, so retrieval misses paraphrases, which shows up as the occasional
  miss in `hit@k`. Switching to `RAG_EMBEDDING_PROVIDER=local` (sentence-transformers) or `openai`
  materially improves retrieval; re-run `make eval` to compare.
- **groundedness** is a lexical-overlap proxy, not a learned faithfulness judge. An optional
  LLM-as-judge mode is a natural extension (see the roadmap).
- The corpus is intentionally tiny so the suite is fast and fully reproducible.
