# ====================================================================
# PROJECT: LLM SUMMARIZATION API (FastAPI + BART-Large)
# DESCRIPTION: Makefile for managing Docker build, environment, 
#              code quality checks (formatting), and tests within the container.
# ====================================================================

SERVICE_NAME = summarization_api 
DOCKER_COMPOSE = docker-compose
SRC_DIRS = app/ model/ tests/

# --------------------------------------------------------------------
# 1. ENVIRONMENT & LIFECYCLE COMMANDS
# --------------------------------------------------------------------

.PHONY: build run down logs shell clean

build:
	@echo "Building Docker image."
	$(DOCKER_COMPOSE) build

run: build
	@echo "Starting service in detached mode."
	$(DOCKER_COMPOSE) up -d

down:
	@echo "Stopping and removing containers."
	$(DOCKER_COMPOSE) down

logs:
	@echo "Following service logs."
	$(DOCKER_COMPOSE) logs -f $(SERVICE_NAME)

shell:
	@echo "Entering service container shell."
	$(DOCKER_COMPOSE) exec $(SERVICE_NAME) sh

clean: down
	@echo "Cleaning up unused Docker images and volumes."
	docker system prune -f --volumes

# --------------------------------------------------------------------
# 2. CODE QUALITY & TESTING COMMANDS
# --------------------------------------------------------------------

.PHONY: quality format test

quality: format

TEST_VOLUMES = -v "$(PWD)/app":/app/app -v "$(PWD)/model":/app/model -v "$(PWD)/tests":/app/tests

format:
	@echo "Running Black Code Formatter."
	$(DOCKER_COMPOSE) run --rm $(TEST_VOLUMES) $(SERVICE_NAME) python -m black $(SRC_DIRS)
	@echo "Running Isort (Import Sorter)."
	$(DOCKER_COMPOSE) run --rm $(TEST_VOLUMES) $(SERVICE_NAME) python -m isort $(SRC_DIRS)

test:
	@echo "Running Pytest Unit Tests."
	$(DOCKER_COMPOSE) run --rm $(TEST_VOLUMES) $(SERVICE_NAME) python -m pytest tests/
