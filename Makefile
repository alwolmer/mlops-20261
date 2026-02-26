.PHONY: help install lint test clean all

# Variables
PYTHON := python3
PIP := pip3
DOCKER_USERNAME ?= $(shell echo $$DOCKER_USERNAME)
IMAGE_NAME := $(DOCKER_USERNAME)/mlops-project

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint:
	$(PYTHON) -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	$(PYTHON) -m flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

test:
	$(PYTHON) -m pytest -v

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +

all: install lint test