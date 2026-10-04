# Sample corpus

This folder holds the small set of documents the demo and the evaluation run against, so the
whole pipeline works on a fresh clone with nothing external to set up.

## What's here

`corpus/` contains eight short Markdown articles on stable, everyday topics: photosynthesis,
the water cycle, TCP vs. UDP, honeybee colonies, Roman aqueducts, the James Webb Space
Telescope, espresso extraction, and the Gutenberg press. They are mostly plain prose, though a
few use a short list or sub-heading. Each has a single `# Title` heading and no links or
images, and the articles don't reference one another, so each can be ingested and retrieved on
its own.

The topics were chosen to be uncontroversial and stable over time, with clear factual answers.
That matters for evaluation: metrics like hit@k and MRR only mean something when each question
has an unambiguous supporting document, which wouldn't hold for opinionated or fast-moving
subjects.

## How it's used

The filesystem loader reads the Markdown files, and each document is cleaned (whitespace and
Markdown normalised), split into overlapping chunks, embedded, and stored in the vector index.
The same articles are the ground truth for evaluation: the questions in
`evaluation/qa_dataset.json` are written against specific facts in these files, so each one
traces back to the article that should be retrieved to answer it.
