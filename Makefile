SHELL := /usr/bin/env bash

FUNCTION_CHAIN_COMPOSE := docker compose -f infra/function-chain-rerank/docker-compose.yml
EMBEDDING_LIST_COMPOSE := docker compose -f infra/embedding-list-max-sim/docker-compose.yml
STRUCTARRAY_COMPOSE := docker compose -f infra/structarray-search/docker-compose.yml
PYTHON_PROJECTS := \
	packages/python/milvus-demo-common \
	demos/function-chain-rerank/backend \
	demos/embedding-list-max-sim/backend \
	demos/structarray-search/backend

.DEFAULT_GOAL := help

.PHONY: help bootstrap format-check lint typecheck test test-python test-web \
	compose-config up down logs \
	function-chain-rerank-data-check \
	embedding-list-max-sim-data-check \
	structarray-search-data-check validate

help:
	@echo "Milvus 3.0 Demos commands"
	@echo "  make bootstrap                          Install development dependencies"
	@echo "  make up / down / logs                   Start, stop, or tail the demo backend"
	@echo "  make validate                           Run fast validation without E2E"
	@echo "  make function-chain-rerank-data-check   Verify the synthetic catalog offline"
	@echo "  make embedding-list-max-sim-data-check  Verify the NASA handbook dataset offline"
	@echo "  make structarray-search-data-check      Validate the approved CoVLA 30-video slice"

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
	$(EMBEDDING_LIST_COMPOSE) config --quiet
	$(STRUCTARRAY_COMPOSE) config --quiet

up:
	$(FUNCTION_CHAIN_COMPOSE) up --build --detach --wait

down:
	$(FUNCTION_CHAIN_COMPOSE) down --remove-orphans

logs:
	$(FUNCTION_CHAIN_COMPOSE) logs --follow --tail=200

function-chain-rerank-data-check:
	UV_OFFLINE=1 uv run --offline --project demos/function-chain-rerank/backend \
		python -m function_chain_demo.dataset validate

embedding-list-max-sim-data-check:
	@set -euo pipefail; \
	dataset_root="demos/embedding-list-max-sim/data/nasa-systems-engineering-handbook-rev2"; \
	temporary_root="$$(mktemp -d)"; \
	trap 'rm -rf "$$temporary_root"' EXIT; \
	UV_OFFLINE=1 uv run --offline --project demos/embedding-list-max-sim/backend \
		python -m embedding_list_demo.manual validate "$$dataset_root"; \
	UV_OFFLINE=1 uv run --offline --project demos/embedding-list-max-sim/backend \
		python -m embedding_list_demo.manual rebuild-check "$$dataset_root" \
		"$$temporary_root/rebuilt"

structarray-search-data-check:
	UV_OFFLINE=1 uv run --offline --project demos/structarray-search/backend \
		python -m structarray_hybrid_demo.cli data-check

validate: function-chain-rerank-data-check embedding-list-max-sim-data-check format-check lint typecheck test compose-config
