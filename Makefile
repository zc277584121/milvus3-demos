SHELL := /usr/bin/env bash

FUNCTION_CHAIN_COMPOSE := docker compose -f infra/function-chain-rerank/docker-compose.yml
STRUCTARRAY_COMPOSE := docker compose -f infra/structarray-search/docker-compose.yml
EMBEDDING_LIST_COMPOSE := docker compose -f infra/embedding-list-max-sim/docker-compose.yml
PYTHON_PROJECTS := \
	demos/function-chain-rerank/backend \
	demos/structarray-search/backend \
	demos/embedding-list-max-sim/backend

.DEFAULT_GOAL := help

.PHONY: help bootstrap format-check lint typecheck test test-python test-web \
	compose-config \
	function-chain-rerank-data-check function-chain-rerank-image-check \
	structarray-search-data-check structarray-search-image-check \
	embedding-list-max-sim-data-check embedding-list-max-sim-image-check \
	validate

help:
	@echo "Milvus 3.0 Demos commands"
	@echo "  make bootstrap             Install development dependencies"
	@echo "  make validate              Run fast validation without container checks"
	@echo "  make function-chain-rerank-data-check   Verify the synthetic commerce catalog offline"
	@echo "  make function-chain-rerank-image-check  Build and health-check the isolated rerank image"
	@echo "  make structarray-search-data-check      Validate synthetic driving data"
	@echo "  make structarray-search-image-check     Build and health-check the StructArray image"
	@echo "  make embedding-list-max-sim-data-check  Rebuild and byte-check NASA PDF pages"
	@echo "  make embedding-list-max-sim-image-check Build and health-check the ColSmol image"

bootstrap:
	@set -euo pipefail; for project in $(PYTHON_PROJECTS); do uv sync --project "$$project" --group dev; done
	npm install

format-check:
	uv run --project demos/function-chain-rerank/backend --group dev ruff format --check demos
	npm run format:check

lint:
	uv run --project demos/function-chain-rerank/backend --group dev ruff check demos

typecheck:
	npm run typecheck

test-python:
	@set -euo pipefail; for project in $(PYTHON_PROJECTS); do uv run --project "$$project" --group dev pytest "$$project/tests"; done

test-web:
	npm test

test: test-python test-web

compose-config:
	$(FUNCTION_CHAIN_COMPOSE) config --quiet
	$(STRUCTARRAY_COMPOSE) config --quiet
	$(EMBEDDING_LIST_COMPOSE) config --quiet

function-chain-rerank-data-check:
	UV_OFFLINE=1 uv run --offline --project demos/function-chain-rerank/backend \
		python -m function_chain_demo.dataset validate

function-chain-rerank-image-check:
	@set -euo pipefail; \
	test -n "$${HF_HUB_CACHE_HOST:-}" || { echo "HF_HUB_CACHE_HOST is required" >&2; exit 1; }; \
	trap '$(FUNCTION_CHAIN_COMPOSE) down --remove-orphans' EXIT; \
	$(FUNCTION_CHAIN_COMPOSE) build --no-cache; \
	$(FUNCTION_CHAIN_COMPOSE) up --detach --wait; \
	curl --fail --silent --show-error "http://127.0.0.1:$${FUNCTION_CHAIN_PORT:-48020}/healthz/ready"

structarray-search-data-check:
	UV_OFFLINE=1 uv run --offline \
		--project demos/structarray-search/backend \
		python -m structarray_hybrid_demo.cli data-check

structarray-search-image-check:
	@set -euo pipefail; \
	trap '$(STRUCTARRAY_COMPOSE) down --remove-orphans' EXIT; \
	$(STRUCTARRAY_COMPOSE) build --no-cache; \
	$(STRUCTARRAY_COMPOSE) up --detach --wait; \
	curl --fail --silent --show-error "http://127.0.0.1:$${STRUCTARRAY_PORT:-48030}/healthz/live"

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

embedding-list-max-sim-image-check:
	@set -euo pipefail; \
	test -n "$${HF_HUB_CACHE_HOST:-}" || { echo "HF_HUB_CACHE_HOST is required" >&2; exit 1; }; \
	trap '$(EMBEDDING_LIST_COMPOSE) down --remove-orphans' EXIT; \
	$(EMBEDDING_LIST_COMPOSE) build --no-cache; \
	$(EMBEDDING_LIST_COMPOSE) up --detach --wait; \
	curl --fail --silent --show-error "http://127.0.0.1:$${EMBEDDING_LIST_PORT:-48040}/healthz/live"

validate: embedding-list-max-sim-data-check format-check lint typecheck test compose-config
