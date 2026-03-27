# ============================================================
# Scientific Document Intelligence Pipeline — Makefile
# ============================================================
# Single-command interface for the entire pipeline.
#
# QUICK START:
#   make setup          → First-time setup (copy .env, create dirs)
#   make build          → Build Docker images
#   make up             → Start all services (ChromaDB + Neo4j)
#   make run            → Process all documents in data/input/
#   make run FILE=...   → Process a single file
#   make run URL=...    → Process a web URL
#   make search Q="..." → Semantic search over indexed documents
#   make stats          → Show pipeline statistics
#   make down           → Stop all services
# ============================================================

# ─── Configuration Variables (overridable via CLI or .env) ────
SHELL         := /bin/bash
.DEFAULT_GOAL := help

# Docker compose command
DC            := docker compose

# Project name (used for container naming)
PROJECT       := sdip

# Pipeline service name in docker-compose
PIPELINE_SVC  := pipeline-shell

# Default input/output directories
INPUT_DIR     := data/input
OUTPUT_DIR    := data/output

# Default pipeline mode (full | extract_only | search_only | graph_only)
MODE          ?= full

# Default schema (default | research_paper | patent)
SCHEMA        ?= default

# Log level
LOG_LEVEL     ?= INFO

# Search top-k results
TOP_K         ?= 10

# File / URL / Directory for processing
FILE          ?=
URL           ?=
DIR           ?= $(INPUT_DIR)

# Query for semantic search
Q             ?=

# Glob pattern for directory processing
GLOB          ?= **/*

# ─── Colors ───────────────────────────────────────────────────
CYAN    := \033[36m
GREEN   := \033[32m
YELLOW  := \033[33m
RED     := \033[31m
BOLD    := \033[1m
RESET   := \033[0m

# ═══════════════════════════════════════════════════════════════
# SETUP & INITIALIZATION
# ═══════════════════════════════════════════════════════════════

.PHONY: setup
setup: ## First-time project setup
	@echo -e "$(BOLD)$(CYAN)Setting up Scientific Document Intelligence Pipeline...$(RESET)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo -e "$(YELLOW)⚠  Created .env from .env.example — add your OPENAI_API_KEY$(RESET)"; \
	else \
		echo -e "$(GREEN)✓  .env already exists$(RESET)"; \
	fi
	@mkdir -p $(INPUT_DIR) $(OUTPUT_DIR)
	@echo -e "$(GREEN)✓  Created data directories: $(INPUT_DIR)/ $(OUTPUT_DIR)/$(RESET)"
	@echo -e "$(GREEN)✓  Setup complete! Next: $(BOLD)make build && make up$(RESET)"


# ═══════════════════════════════════════════════════════════════
# DOCKER BUILD & LIFECYCLE
# ═══════════════════════════════════════════════════════════════

.PHONY: build
build: ## Build the pipeline Docker image
	@echo -e "$(BOLD)$(CYAN)Building Docker image...$(RESET)"
	$(DC) build --no-cache pipeline-shell
	@echo -e "$(GREEN)✓  Build complete$(RESET)"

.PHONY: build-fast
build-fast: ## Build without --no-cache (uses Docker layer cache)
	@echo -e "$(CYAN)Building (cached)...$(RESET)"
	$(DC) build pipeline-shell

.PHONY: up
up: ## Start all background services (ChromaDB + Neo4j)
	@echo -e "$(BOLD)$(CYAN)Starting services...$(RESET)"
	$(DC) up -d chromadb neo4j
	@echo -e "$(CYAN)Waiting for services to be healthy...$(RESET)"
	@$(MAKE) _wait-healthy
	@echo -e "$(GREEN)✓  Services ready$(RESET)"
	@echo -e "   ChromaDB: $(CYAN)http://localhost:8000$(RESET)"
	@echo -e "   Neo4j:    $(CYAN)http://localhost:7474$(RESET)"

.PHONY: down
down: ## Stop and remove all containers
	@echo -e "$(CYAN)Stopping services...$(RESET)"
	$(DC) down
	@echo -e "$(GREEN)✓  Services stopped$(RESET)"

.PHONY: down-volumes
down-volumes: ## Stop services and remove all persistent volumes (DESTRUCTIVE)
	@echo -e "$(RED)$(BOLD)⚠  This will delete all stored data (ChromaDB + Neo4j volumes)$(RESET)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	$(DC) down -v
	@echo -e "$(GREEN)✓  Services and volumes removed$(RESET)"

.PHONY: restart
restart: ## Restart all services
	@$(MAKE) down
	@$(MAKE) up

.PHONY: logs
logs: ## Tail logs from all services
	$(DC) logs -f

.PHONY: logs-pipeline
logs-pipeline: ## Tail pipeline container logs
	$(DC) logs -f pipeline-shell

.PHONY: logs-chromadb
logs-chromadb: ## Tail ChromaDB logs
	$(DC) logs -f chromadb

.PHONY: logs-neo4j
logs-neo4j: ## Tail Neo4j logs
	$(DC) logs -f neo4j

.PHONY: ps
ps: ## Show running container status
	$(DC) ps


# ═══════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════

.PHONY: run
run: _check-env ## Run pipeline (FILE=, URL=, DIR=, or default data/input/)
	@echo -e "$(BOLD)$(CYAN)Running pipeline | mode=$(MODE) | schema=$(SCHEMA)$(RESET)"
	$(DC) run --rm \
		-e MODE=$(MODE) \
		-e SCHEMA=$(SCHEMA) \
		-e LOG_LEVEL=$(LOG_LEVEL) \
		$(PIPELINE_SVC) \
		$(call _build-run-cmd,$(FILE),$(URL),$(DIR))

.PHONY: run-local
run-local: _check-local-env ## Run pipeline locally (without Docker)
	@echo -e "$(BOLD)$(CYAN)Running pipeline locally | mode=$(MODE)$(RESET)"
	PYTHONPATH=. python -m src.main run \
		$(call _build-local-run-args,$(FILE),$(URL),$(DIR)) \
		--mode $(MODE) \
		--schema $(SCHEMA) \
		--log-level $(LOG_LEVEL)

.PHONY: search
search: _check-env ## Semantic search (Q="your query" [TOP_K=10])
	@if [ -z "$(Q)" ]; then \
		echo -e "$(RED)Error: Q is required. Usage: make search Q=\"your query\"$(RESET)"; \
		exit 1; \
	fi
	@echo -e "$(CYAN)Searching: $(Q)$(RESET)"
	$(DC) run --rm $(PIPELINE_SVC) \
		python -m src.main search "$(Q)" --top-k $(TOP_K)

.PHONY: stats
stats: ## Show pipeline statistics (indexed docs, graph size, outputs)
	$(DC) run --rm $(PIPELINE_SVC) \
		python -m src.main stats

.PHONY: clear-outputs
clear-outputs: ## Clear output JSON files only
	$(DC) run --rm $(PIPELINE_SVC) \
		python -m src.main clear --outputs

.PHONY: clear-all
clear-all: ## Clear ChromaDB + Neo4j + outputs (DESTRUCTIVE)
	@echo -e "$(RED)$(BOLD)⚠  This will delete ALL indexed data$(RESET)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	$(DC) run --rm $(PIPELINE_SVC) \
		python -m src.main clear --all


# ═══════════════════════════════════════════════════════════════
# DEVELOPMENT
# ═══════════════════════════════════════════════════════════════

.PHONY: shell
shell: ## Open an interactive shell in the pipeline container
	@echo -e "$(CYAN)Opening pipeline shell...$(RESET)"
	$(DC) run --rm -it $(PIPELINE_SVC) /bin/bash

.PHONY: shell-python
shell-python: ## Open a Python REPL in the pipeline container
	$(DC) run --rm -it $(PIPELINE_SVC) python

.PHONY: test
test: ## Run all tests
	@echo -e "$(BOLD)$(CYAN)Running tests...$(RESET)"
	$(DC) run --rm $(PIPELINE_SVC) \
		python -m pytest tests/ -v --tb=short
	@echo -e "$(GREEN)✓  Tests complete$(RESET)"

.PHONY: test-local
test-local: ## Run tests locally without Docker
	PYTHONPATH=. python -m pytest tests/ -v --tb=short

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	$(DC) run --rm $(PIPELINE_SVC) \
		python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html:data/output/coverage

.PHONY: lint
lint: ## Run linting (ruff + mypy)
	$(DC) run --rm $(PIPELINE_SVC) bash -c "ruff check src/ && mypy src/ --ignore-missing-imports"

.PHONY: format
format: ## Auto-format code with ruff
	$(DC) run --rm $(PIPELINE_SVC) ruff format src/ tests/

.PHONY: neo4j-browser
neo4j-browser: ## Print Neo4j browser URL
	@echo -e "$(CYAN)Neo4j Browser: $(BOLD)http://localhost:7474$(RESET)"
	@echo -e "Username: neo4j | Password: (from .env NEO4J_PASSWORD)"

.PHONY: chromadb-ui
chromadb-ui: ## Print ChromaDB API URL
	@echo -e "$(CYAN)ChromaDB API: $(BOLD)http://localhost:8000$(RESET)"
	@echo -e "Collections:  $(CYAN)http://localhost:8000/api/v1/collections$(RESET)"


# ═══════════════════════════════════════════════════════════════
# PIPELINE MODES (convenience shortcuts)
# ═══════════════════════════════════════════════════════════════

.PHONY: run-full
run-full: ## Run full pipeline (extract + embed + graph + summarize)
	@$(MAKE) run MODE=full FILE=$(FILE) URL=$(URL) DIR=$(DIR)

.PHONY: run-extract
run-extract: ## Run extraction only (no embedding or graph)
	@$(MAKE) run MODE=extract_only FILE=$(FILE) URL=$(URL) DIR=$(DIR)

.PHONY: run-search
run-search: ## Run embedding/indexing only (for semantic search)
	@$(MAKE) run MODE=search_only FILE=$(FILE) URL=$(URL) DIR=$(DIR)

.PHONY: run-graph
run-graph: ## Run knowledge graph construction only
	@$(MAKE) run MODE=graph_only FILE=$(FILE) URL=$(URL) DIR=$(DIR)

.PHONY: run-paper
run-paper: ## Process as research paper (research_paper schema)
	@$(MAKE) run SCHEMA=research_paper FILE=$(FILE) URL=$(URL) DIR=$(DIR)

.PHONY: run-patent
run-patent: ## Process as patent document (patent schema)
	@$(MAKE) run SCHEMA=patent FILE=$(FILE) URL=$(URL) DIR=$(DIR)


# ═══════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════

.PHONY: _check-env
_check-env:
	@if [ ! -f .env ]; then \
		echo -e "$(RED)Error: .env file not found. Run: make setup$(RESET)"; exit 1; \
	fi
	@if [ -z "$(OPENAI_API_KEY)" ]; then \
		if ! grep -q "^OPENAI_API_KEY=sk-" .env 2>/dev/null; then \
			echo -e "$(YELLOW)⚠  OPENAI_API_KEY not set in .env — API calls will fail$(RESET)"; \
		fi \
	fi

.PHONY: _check-local-env
_check-local-env:
	@if [ -z "$$OPENAI_API_KEY" ]; then \
		echo -e "$(RED)Error: OPENAI_API_KEY environment variable not set$(RESET)"; exit 1; \
	fi

.PHONY: _wait-healthy
_wait-healthy:
	@echo -n "   Waiting for ChromaDB"
	@for i in $$(seq 1 30); do \
		if curl -sf http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1; then \
			echo -e " $(GREEN)✓$(RESET)"; break; \
		fi; \
		echo -n "."; sleep 2; \
	done
	@echo -n "   Waiting for Neo4j"
	@for i in $$(seq 1 40); do \
		if curl -sf http://localhost:7474 > /dev/null 2>&1; then \
			echo -e " $(GREEN)✓$(RESET)"; break; \
		fi; \
		echo -n "."; sleep 3; \
	done

# Build the run command based on which input args are set
define _build-run-cmd
$(if $(1), \
	python -m src.main run --file $(1) --mode $(MODE) --schema $(SCHEMA), \
$(if $(2), \
	python -m src.main run --url $(2) --mode $(MODE) --schema $(SCHEMA), \
	python -m src.main run --dir $(3) --mode $(MODE) --schema $(SCHEMA) \
))
endef

# Local run args (no docker)
define _build-local-run-args
$(if $(1),--file $(1),$(if $(2),--url $(2),--dir $(3)))
endef


# ═══════════════════════════════════════════════════════════════
# HELP
# ═══════════════════════════════════════════════════════════════

.PHONY: help
help: ## Show this help
	@echo ""
	@echo -e "$(BOLD)$(CYAN)Scientific Document Intelligence Pipeline$(RESET)"
	@echo -e "$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	@echo ""
	@echo -e "$(BOLD)Quick Start:$(RESET)"
	@echo -e "  $(GREEN)make setup$(RESET)                   First-time setup"
	@echo -e "  $(GREEN)make build$(RESET)                   Build Docker image"
	@echo -e "  $(GREEN)make up$(RESET)                      Start ChromaDB + Neo4j"
	@echo -e "  $(GREEN)make run$(RESET)                     Process all docs in data/input/"
	@echo -e "  $(GREEN)make run FILE=doc.pdf$(RESET)        Process a single file"
	@echo -e "  $(GREEN)make run URL=https://...$(RESET)     Process a web URL"
	@echo -e "  $(GREEN)make search Q=\"attention is all you need\"$(RESET)  Semantic search"
	@echo -e "  $(GREEN)make stats$(RESET)                   Show pipeline stats"
	@echo -e "  $(GREEN)make down$(RESET)                    Stop all services"
	@echo ""
	@echo -e "$(BOLD)Variables:$(RESET)"
	@echo -e "  $(YELLOW)FILE$(RESET)      = path to document   (default: none)"
	@echo -e "  $(YELLOW)URL$(RESET)       = web URL            (default: none)"
	@echo -e "  $(YELLOW)DIR$(RESET)       = document directory (default: data/input/)"
	@echo -e "  $(YELLOW)MODE$(RESET)      = pipeline mode      (default: full)"
	@echo -e "              full | extract_only | search_only | graph_only"
	@echo -e "  $(YELLOW)SCHEMA$(RESET)    = extraction schema  (default: default)"
	@echo -e "              default | research_paper | patent"
	@echo -e "  $(YELLOW)Q$(RESET)         = search query"
	@echo -e "  $(YELLOW)TOP_K$(RESET)     = search results     (default: 10)"
	@echo -e "  $(YELLOW)LOG_LEVEL$(RESET) = INFO|DEBUG|WARNING (default: INFO)"
	@echo ""
	@echo -e "$(BOLD)All targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""
