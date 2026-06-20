.DEFAULT_GOAL := help
.PHONY: help setup run cli test lint format check clean docker-build docker-run docker-up docker-down

FILE     ?=
START    ?= 0
DURATION ?= 60
MODEL    ?= small
FORMATS  ?= txt srt json vtt

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install all deps (uv); copies .env from example if missing
	uv sync
	@test -f .env || (cp .env.example .env && echo "Created .env — set HF_TOKEN before running.")

run: setup ## Launch the Streamlit UI at http://localhost:8501
	uv run streamlit run app.py

cli: setup ## Quick fragment test, e.g. make cli FILE=sample-interview.mp3 DURATION=60 MODEL=small
	@test -n "$(FILE)" || (echo "Set FILE=... (path to an audio/video file)"; exit 1)
	uv run transcribe --input "$(FILE)" --start $(START) --duration $(DURATION) \
		--model $(MODEL) --formats $(FORMATS)

test: ## Run unit tests
	uv run pytest

lint: ## Lint with ruff
	uv run ruff check .

format: ## Format with ruff
	uv run ruff format .

check: lint test ## Lint + test

clean: ## Remove caches and the virtualenv
	rm -rf .venv .pytest_cache .ruff_cache **/__pycache__

docker-build: ## Build the Docker image
	docker compose build

docker-run: docker-up ## Alias for docker-up

docker-up: ## Build (if needed) and start the app in Docker at http://localhost:8501
	docker compose up --build

docker-down: ## Stop and remove the Docker container
	docker compose down
