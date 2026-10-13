.DEFAULT_GOAL := help
.PHONY: help quickstart setup run cli test lint format check clean docker-up docker-down \
	install-parakeet install-parakeet-mlx install-qwen3-asr install-sensevoice install-all run-all

# Shortcuts for macOS and Linux. Windows users: run.bat, or the uv commands in README.md.

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

# Every optional engine. Parakeet runs on ONNX without an extra; on Apple Silicon
# its MLX backend is added because it is light and uses the GPU. NeMo (the CUDA
# backend, several GB) is left to `make install-parakeet`.
ALL_EXTRAS := qwen3-asr sensevoice
ifeq ($(shell uname -sm),Darwin arm64)
ALL_EXTRAS += parakeet-mlx
endif

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

quickstart: ## Whisper-only UI with nothing to set up (same as ./run.sh)
	./run.sh

setup: ## Create .venv with all core deps; add EXTRAS=qwen3-asr to include an engine
	uv sync --inexact $(SYNC_EXTRAS)
	@test -f .env || (cp .env.example .env && echo "Created .env — set HF_TOKEN for speaker identification.")

run: setup ## Full UI with speaker identification at http://localhost:8501
	uv run --no-sync streamlit run app.py

install-parakeet: ## Optional Parakeet backend: NVIDIA NeMo, for CUDA GPUs (several GB)
	uv sync --inexact --extra parakeet

install-parakeet-mlx: ## Optional Parakeet backend: MLX, for the Apple Silicon GPU
	uv sync --inexact --extra parakeet-mlx

install-qwen3-asr: ## Install the Qwen3-ASR engine
	uv sync --inexact --extra qwen3-asr

install-sensevoice: ## Install the SenseVoice engine
	uv sync --inexact --extra sensevoice

install-all: ## Install all ASR engines at once (heavy; deps may conflict — see pyproject.toml)
	uv sync --inexact $(foreach e,$(ALL_EXTRAS),--extra $(e))

run-all: EXTRAS = $(ALL_EXTRAS)
run-all: run ## Launch the UI with every ASR engine installed

cli: setup ## Quick fragment test, e.g. make cli FILE=interview.mp3 DURATION=60 MODEL=small
	@test -n "$(FILE)" || (echo "Set FILE=... (path to an audio/video file)"; exit 1)
	uv run --no-sync transcribe --input "$(FILE)" --start $(START) --duration $(DURATION) \
		--model $(MODEL) --formats $(FORMATS)

test: ## Run unit tests
	uv run pytest

lint: ## Lint and check formatting with ruff
	uv run ruff check .
	uv run ruff format --check .

format: ## Format with ruff
	uv run ruff check --fix .
	uv run ruff format .

check: lint test ## Lint + test

clean: ## Remove caches, virtualenvs and generated previews (keeps downloaded models)
	rm -rf .venv .venv-whisper .cache .pytest_cache .ruff_cache static
	find . -name __pycache__ -type d -prune -not -path "./.git/*" -exec rm -rf {} +

docker-up: ## Build (if needed) and start the app in Docker at http://localhost:8501
	docker compose up --build

docker-down: ## Stop and remove the Docker container
	docker compose down
