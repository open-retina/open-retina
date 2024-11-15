
test-types:
	export MPYPATH="."; mypy --config scripts/tests/mypy.ini openretina/ scripts/ tests/

test-codestyle:
	ruff check --config scripts/tests/ruff.toml openretina/ scripts/ tests/

test-formatting:
	ruff format --check --diff --config scripts/tests/ruff.toml openretina/ scripts/ tests/

test-unittests:
	pytest tests/

test-all: test-types test-codestyle test-formatting test-unittests

fix-codestyle:
	ruff check --config scripts/tests/ruff.toml openretina/ scripts/ tests/ --fix

fix-formatting:
	ruff format --config scripts/tests/ruff.toml openretina/ scripts/ tests/

fix-all: fix-codestyle fix-formatting
 
