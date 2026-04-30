FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

# -----------------------------
# Builder: install aegra-cli at the pinned version
# -----------------------------
FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY aegra-cli-version.txt ./
RUN PIN="$(tr -d '[:space:]' < aegra-cli-version.txt)" && \
    uv tool install --force "aegra-cli==${PIN#v}" \
        --with langchain-openai \
        --with langchain-ollama \
        --with langchain-mcp-adapters \
        --with langfuse

# -----------------------------
# Runtime
# -----------------------------
FROM base AS final

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /home/app/.local
RUN chown -R app:app /home/app/.local

COPY aegra.json ./
COPY graphs ./graphs

ENV PATH="/home/app/.local/bin:$PATH"

EXPOSE 8000

USER app

CMD ["aegra", "serve", "--config", "aegra.json", "--host", "0.0.0.0", "--port", "8000"]
