.PHONY: test

test:
	pytest

pretty:
	ruff format
