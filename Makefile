.PHONY: help install dev test

help:
	@echo "make install   Install latest aegra-cli + dev deps"
	@echo "make dev       Run 'aegra dev' against ./aegra.json on port 4242"
	@echo "make test      Run pytest"

install:
	uv tool install --force aegra-cli \
		--with langchain-openai --with langchain-ollama \
		--with langchain-mcp-adapters --with langfuse
	uv sync

dev:
	uv tool run --from aegra-cli \
		--with langchain-openai --with langchain-ollama \
		--with langchain-mcp-adapters --with langfuse \
		aegra dev --config aegra.json --port 4242

test:
	uv run pytest
