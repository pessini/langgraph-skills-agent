.PHONY: help backend-cli-install backend-up backend-down format lint test clean

help:
	@echo "Available commands:"
	@echo "  make backend-cli-install - Install pinned Aegra CLI"
	@echo "  make backend-up          - Run 'aegra dev' with aegra.json"
	@echo "  make backend-down        - Stop the dev backend (Ctrl-C also works)"
	@echo "  make test                - Run pytest"
	@echo "  make lint                - Lint with ruff"
	@echo "  make format              - Format with ruff"
	@echo "  make clean               - Clean cache files"

backend-cli-install:
	./scripts/install-aegra-cli.sh

backend-up:
	uv tool run --from aegra-cli \
		--with langchain-openai \
		--with langchain-ollama \
		--with langchain-mcp-adapters \
		--with langfuse \
		aegra dev --config aegra.json --port 4242

backend-down:
	@echo "Stop the foreground 'aegra dev' process with Ctrl-C."

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov 2>/dev/null || true
