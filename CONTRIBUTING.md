# Contributing

Thanks for your interest! This project uses [uv](https://docs.astral.sh/uv/) for everything.

## Development setup

```bash
uv sync --dev --extra ui --extra chroma --extra qdrant --extra otel   # or: make setup
uv run pre-commit install
```

Optional extras: `--extra local` adds sentence-transformers (pulls in torch); `--group media`
adds Playwright/Pillow for regenerating UI screenshots.

## Day-to-day

| Command | What it does |
| --- | --- |
| `make fmt` | Auto-fix lint issues (ruff) and format (black) |
| `make lint` | Lint + formatting check (CI gate) |
| `make type` | `mypy --strict` |
| `make test` | Run the test suite |
| `make test-cov` | Tests with coverage (must stay ≥ 80%) |
| `make e2e` | End-to-end test over the sample corpus |
| `make run` / `make ui` | Run the API / the Streamlit UI locally |
| `make eval` | Regenerate the evaluation results |

## Standards

- **Python 3.11+**, full type hints. `ruff`, `black`, and `mypy --strict` must pass (they run
  in pre-commit and CI).
- The domain core (`rag.domain`) must not import a framework or a provider SDK. New
  providers/stores are adapters behind the existing ports, wired in `rag.factory`.
- Every non-trivial function gets a short usage example (docstring) and a test. Keep test
  coverage at or above the 80% gate.
- **No secrets in code or fixtures.** Read configuration through `Settings`.

## Pull requests

1. Branch from `main`.
2. Make focused commits ([Conventional Commits](https://www.conventionalcommits.org/) style).
3. Ensure `make lint type test-cov` is green locally.
4. Open a PR describing the change and how you verified it; CI must pass.
