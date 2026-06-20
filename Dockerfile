FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_LINK_MODE=copy \
    HF_HOME=/app/hf_cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

COPY pyproject.toml uv.lock* README.md ./
COPY src ./src
RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

COPY app.py ./
COPY .streamlit ./.streamlit

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py"]
