.PHONY: test

test:
	pytest

pretty:
	black *.py
