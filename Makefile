.DEFAULT_GOAL := help
.PHONY: help setup clean-static run cli test lint format check clean docker-build docker-run docker-up docker-down \
	install-parakeet install-parakeet-mlx install-qwen3-asr install-sensevoice install-all run-all

FILE     ?=
START    ?= 0
DURATION ?= 60
MODEL    ?= small
FORMATS  ?= txt srt json vtt
EXTRAS   ?=

# Optional ASR backends to sync, e.g. make run EXTRAS=qwen3-asr
# Space-separate to enable several, but note they may conflict (see pyproject.toml).
# Recursively expanded (=) so target-specific EXTRAS (e.g. run-all) reaches setup.
SYNC_EXTRAS = $(foreach e,$(EXTRAS),--extra $(e))

# Every ASR engine. Apple Silicon uses the MLX Parakeet backend, others NeMo.
ALL_EXTRAS := qwen3-asr sensevoice
ifeq ($(shell uname -sm),Darwin arm64)
ALL_EXTRAS += parakeet-mlx
else
ALL_EXTRAS += parakeet
endif

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install all deps (uv); copies .env from example if missing. Add EXTRAS=qwen3-asr to include an ASR backend
	uv sync $(SYNC_EXTRAS)
	@test -f .env || (cp .env.example .env && echo "Created .env — set HF_TOKEN before running.")

install-parakeet: ## Install the Parakeet engine (NVIDIA NeMo; use install-parakeet-mlx on Apple Silicon)
	uv sync --extra parakeet

install-parakeet-mlx: ## Install the Parakeet engine (Apple Silicon MLX backend)
	uv sync --extra parakeet-mlx

install-qwen3-asr: ## Install the Qwen3-ASR engine
	uv sync --extra qwen3-asr

install-sensevoice: ## Install the SenseVoice engine
	uv sync --extra sensevoice

install-all: ## Install all ASR engines at once (heavy; deps may conflict — see pyproject.toml)
	uv sync $(foreach e,$(ALL_EXTRAS),--extra $(e))

clean-static: ## Remove generated Streamlit preview files
	rm -f static/preview_*

run: setup clean-static ## Launch the Streamlit UI at http://localhost:8501
	uv run streamlit run app.py

run-all: EXTRAS = $(ALL_EXTRAS)
run-all: run ## Launch the UI with every ASR engine installed (one uv sync; deps may conflict)

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
	rm -rf .venv .pytest_cache .ruff_cache **/__pycache__ static/preview_*

docker-build: ## Build the Docker image
	docker compose build

docker-run: docker-up ## Alias for docker-up

docker-up: ## Build (if needed) and start the app in Docker at http://localhost:8501
	docker compose up --build

docker-down: ## Stop and remove the Docker container
	docker compose down
