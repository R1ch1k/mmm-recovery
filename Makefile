# mmm-recovery
#
# PREREGISTRATION.md is the binding spec; CLAUDE.md holds the build order.
# `reproduce` and `report` intentionally point at modules that do not exist yet —
# they land at Steps 6 and 8. Every step must leave `make test` and `make lint` green.

.PHONY: help install test lint reproduce report
.DEFAULT_GOAL := help

UV := uv run

help:
	@echo "install    sync the locked environment (core deps + dev group)"
	@echo "test       run the test suite"
	@echo "lint       ruff check, ruff format --check, mypy --strict"
	@echo "reproduce  run the full condition grid -> results/grid.csv"
	@echo "report     build the self-contained dashboard -> results/dashboard.html"

install:
	uv sync

test:
	$(UV) pytest

lint:
	$(UV) ruff check .
	$(UV) ruff format --check .
	$(UV) mypy

reproduce:
	$(UV) python -m mmm_recovery.experiment

report:
	$(UV) python -m mmm_recovery.report
