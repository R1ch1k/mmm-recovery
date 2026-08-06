# mmm-recovery
#
# PREREGISTRATION.md is the binding spec; CLAUDE.md holds the build order.
# `reproduce` still points at a module that does not exist: the kill criterion K1 fired on C0
# before Step 6, so no degradation grid was ever run and `experiment.py` was never written.
# That is a recorded outcome (D21, D23), not an omission. Every step must leave `make test`
# and `make lint` green.

.PHONY: help install test lint reproduce plateau report
.DEFAULT_GOAL := help

UV := uv run

help:
	@echo "install    sync the locked environment (core deps + dev group)"
	@echo "test       run the test suite"
	@echo "lint       ruff check, ruff format --check, mypy --strict"
	@echo "reproduce  blocked by K1 -- experiment.py was never written (D21, D23)"
	@echo "plateau    regenerate the identification plateau -> results/plateau_sweep.csv"
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

plateau:
	$(UV) python -m mmm_recovery.plateau

# plotly is the [report] extra: it is vendored into the HTML at build time and is never a
# run-time dependency of the grid.
report:
	uv run --extra report python -m mmm_recovery.report
