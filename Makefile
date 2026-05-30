.PHONY: help install dev test

help:
	@echo "make install   Install project dependencies"
	@echo "make dev       Run 'langgraph dev' against ./langgraph.json on port 2024"
	@echo "make test      Run pytest"

install:
	uv sync

dev:
	uv run --with 'langgraph-cli[inmem]' langgraph dev --config langgraph.json --port 2024

test:
	uv run pytest
