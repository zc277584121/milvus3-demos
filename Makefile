SHELL := /usr/bin/env bash

FUNCTION_CHAIN_COMPOSE := docker compose -f infra/function-chain-rerank/docker-compose.yml
PYTHON_PROJECTS := \
	packages/python/milvus-demo-common \
	demos/function-chain-rerank/backend

.DEFAULT_GOAL := help

.PHONY: help bootstrap format-check lint typecheck test test-python test-web \
	compose-config up down logs function-chain-rerank-data-check validate

help:
	@echo "Milvus 3.0 Function Chain Rerank demo commands"
	@echo "  make bootstrap                        Install development dependencies"
	@echo "  make up / down / logs                 Start, stop, or tail the demo backend"
	@echo "  make validate                         Run fast validation without E2E"
	@echo "  make function-chain-rerank-data-check Verify the synthetic catalog offline"

bootstrap:
	@set -euo pipefail; for project in $(PYTHON_PROJECTS); do uv sync --project "$$project" --group dev; done
	npm install

format-check:
	uv run --project demos/function-chain-rerank/backend --group dev ruff format --check packages/python demos
	npm run format:check

lint:
	uv run --project demos/function-chain-rerank/backend --group dev ruff check packages/python demos

typecheck:
	npm run typecheck

test-python:
	@set -euo pipefail; for project in $(PYTHON_PROJECTS); do uv run --project "$$project" --group dev pytest "$$project/tests"; done

test-web:
	npm test

test: test-python test-web

compose-config:
	$(FUNCTION_CHAIN_COMPOSE) config --quiet

up:
	$(FUNCTION_CHAIN_COMPOSE) up --build --detach --wait

down:
	$(FUNCTION_CHAIN_COMPOSE) down --remove-orphans

logs:
	$(FUNCTION_CHAIN_COMPOSE) logs --follow --tail=200

function-chain-rerank-data-check:
	UV_OFFLINE=1 uv run --offline --project demos/function-chain-rerank/backend \
		python -m function_chain_demo.dataset validate

validate: function-chain-rerank-data-check format-check lint typecheck test compose-config
