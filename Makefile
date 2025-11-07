# # Makefile

# # Variables
# ENV_NAME = protest_safety
# PYTHON = python
# PIP = pip

# # Default target (run when you just type `make`)
# .DEFAULT_GOAL := help

# # Help target
# help:
# 	@echo "Available commands:"
# 	@echo "  make setup        - Create the Conda environment and install packages"
# 	@echo "  make test         - Run the test suite"
# 	@echo "  make lint         - Run the linter"
# 	@echo "  make run          - Execute the main MVP script"
# 	@echo "  make docker-build - Build the Docker image"
# 	@echo "  make sync-deps    - Sync environment.yml and requirements.txt"

# # Setup target
# setup:
# 	conda env create -f environment.yml || conda env update -f environment.yml --prune

# # Test target
# test:
# 	conda run -n $(ENV_NAME) $(PYTHON) -m pytest tests/ -v

# # Lint target
# lint:
# 	conda run -n $(ENV_NAME) flake8 src/ --count --max-line-length=127

# # Run target
# run:
# 	conda run -n $(ENV_NAME) $(PYTHON) src/run_mvp.py

# # Docker build target
# docker-build:
# 	docker build -t $(ENV_NAME) .

# # Sync dependencies target
# sync-deps:
# 	conda activate $(ENV_NAME) && \
# 	conda env export --from-history | grep -v "prefix:" > environment.yml && \
# 	pip freeze > requirements.txt

# .PHONY: help setup test lint run docker-build sync-deps