SHELL := /bin/bash

help:
	@echo "setup - setup pyenv and pipenv"
	@echo "data - get dataset from kaggle"
	@echo "format - format the codebase using Black"

setup:
	bash bins/setup.sh
	pipenv shell

get_data:
	bash bins/get_data.sh

format:
	black .
