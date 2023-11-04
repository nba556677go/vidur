.PHONY: help lint lint/flake8 lint/black lint/isort format format/black format/autopep8 format/isort
.DEFAULT_GOAL := help

lint/flake8: ## check style with flake8
	flake8 simulator

lint/black: ## check style with black
	black --check simulator

lint/isort: ## check style with isort
	isort --check-only --profile black simulator

lint: lint/black lint/isort ## check style

format/black: ## format code with black
	black simulator

format/autopep8: ## format code with autopep8
	autopep8 --in-place --aggressive --aggressive --recursive simulator/

format/isort: ## format code with isort
	isort --profile black simulator

format: format/isort format/black ## format code

run:
	python -m simulator.main

run-with-args:
	python -m simulator.main $(ARGS)

run-with-trace:
	python -m simulator.main --write_chrome_trace True; \
	trace_file=`ls -t simulator_output/*/chrome_trace.json | head -1`; \
	zip -r $$trace_file.zip $$trace_file

run-with-trace-and-args:
	python -m simulator.main --write_chrome_trace True $(ARGS); \
	trace_file=`ls -t simulator_output/*/chrome_trace.json | head -1`; \
	zip -r $$trace_file.zip $$trace_file
