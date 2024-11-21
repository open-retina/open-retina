
test-types:
	mypy openretina/ scripts/ tests/

test-codestyle:
	ruff check openretina/ scripts/ tests/

test-formatting:
	ruff format --check --diff openretina/ scripts/ tests/

test-unittests:
	pytest tests/

test-all: test-types test-codestyle test-formatting test-unittests

fix-codestyle:
	ruff check openretina/ scripts/ tests/ --fix

fix-formatting:
	ruff format openretina/ scripts/ tests/

fix-all: fix-codestyle fix-formatting
 
