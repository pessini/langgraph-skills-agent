AEGRA_CLI_VERSION := 0.7.2

.PHONY: help install dev test

help:
	@echo "make install   Install pinned aegra-cli (v$(AEGRA_CLI_VERSION)) + dev deps"
	@echo "make dev       Run 'aegra dev' against ./aegra.json on port 4242"
	@echo "make test      Run pytest"

install:
	uv tool install --force "aegra-cli==$(AEGRA_CLI_VERSION)" \
		--with langchain-openai --with langchain-ollama \
		--with langchain-mcp-adapters --with langfuse
	uv sync

dev:
	uv tool run --from "aegra-cli==$(AEGRA_CLI_VERSION)" \
		--with langchain-openai --with langchain-ollama \
		--with langchain-mcp-adapters --with langfuse \
		aegra dev --config aegra.json --port 4242

test:
	uv run pytest
