UV ?= uv

.DEFAULT_GOAL := help
.PHONY: help setup fmt lint type test test-cov e2e eval run ui up down build logs security diagrams clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies (offline stack + UI + stores + dev) and pre-commit hooks
	$(UV) sync --dev --extra ui --extra chroma --extra qdrant --extra otel
	$(UV) run pre-commit install || true

fmt: ## Auto-fix lint issues and format the code
	$(UV) run ruff check --fix .
	$(UV) run black .

lint: ## Lint (ruff) and check formatting (black)
	$(UV) run ruff check .
	$(UV) run black --check .

type: ## Strict type-check with mypy
	$(UV) run mypy

test: ## Run the test suite (incl. the optional store adapters)
	$(UV) run --extra chroma --extra qdrant pytest -q

test-cov: ## Run tests with coverage and enforce the 80% gate
	$(UV) run --extra chroma --extra qdrant pytest --cov=rag --cov-report=term-missing --cov-fail-under=80

e2e: ## Run the end-to-end test (ingest sample corpus -> cited answer)
	$(UV) run pytest tests/e2e -q

eval: ## Run the evaluation harness (hit@k, MRR, groundedness)
	$(UV) run python -m evaluation.run_eval

run: ## Run the API locally with autoreload
	$(UV) run uvicorn rag.api.main:app --reload --host 0.0.0.0 --port 8000

ui: ## Run the Streamlit UI locally
	$(UV) run streamlit run ui/streamlit_app.py

up: ## Build and start the full stack (api + ui + qdrant) in the background
	docker compose up --build -d

down: ## Stop the stack and remove containers
	docker compose down

build: ## Build the Docker image
	docker compose build

logs: ## Tail the api logs
	docker compose logs -f api

security: ## Audit dependencies for known vulnerabilities
	$(UV) run pip-audit

diagrams: ## Render the Mermaid diagrams to SVG and PNG
	./scripts/render_diagrams.sh

clean: ## Remove caches, coverage and local data
	rm -rf .data .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
