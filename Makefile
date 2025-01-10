.PHONY: help docs run tests

help:  ## Get a description of what each command does
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

hooks:  ## Install pre-commit hooks
	poetry run pre-commit install

pre-commit: ## Runs the pre-commit over entire repo
	poetry run pre-commit run --all-files

install:  ## Install packages locally using poetry, includes CLI
	poetry install --all-extras --with dev
	
tests:  ## Run all unit tests including ones marked as slow
	poetry run py.test --verbose tests

run-simul:
	poetry run python -m src.alphago.game

calibration:
	poetry run python -m src.experiments.calibration

lmc-connect4:
	poetry run python -m src.experiments.lmc_connect4
