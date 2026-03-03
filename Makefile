.PHONY: install lint format test clean all pipeline-train pipeline-infer

# Variables
PYTHON := python3
UV := uv
UV_RUN := $(UV) run
DOCKER_USERNAME ?= $(shell echo $$DOCKER_USERNAME)
IMAGE_NAME := $(DOCKER_USERNAME)/mlops-project
PIPELINE_INPUT ?= lifecycle/data/raw/risco_credito.csv
PIPELINE_ARTIFACTS ?= lifecycle/data/processed
PIPELINE_OUTPUT ?= lifecycle/data/processed/predictions.csv

install:
	$(UV) sync --frozen --dev --no-install-project

lint:
	$(UV_RUN) ruff check . --fix
	$(UV_RUN) ruff format .

test:
	$(UV_RUN) pytest -v

pipeline-train:
	$(UV_RUN) python -m lifecycle train --input $(PIPELINE_INPUT) --artifacts $(PIPELINE_ARTIFACTS)

pipeline-infer:
	$(UV_RUN) python -m lifecycle infer --input $(PIPELINE_INPUT) --artifacts $(PIPELINE_ARTIFACTS) --output $(PIPELINE_OUTPUT)

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +

all: install lint test
