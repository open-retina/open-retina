
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

test-corereadout:
	./scripts/train_core_readout.py --config-path ~/GitRepos/open-retina/configs --config-name example_train_core_readout_low_res responses_path=https://gin.g-node.org/teulerlab/open-retina/raw/master/responses/eulerlab/rgc_natsim_subset_only_naturalspikes_2024-08-14.h5 +trainer.limit_train_batches=1 trainer.max_epochs=1 +trainer.limit_val_batches=1 +trainer.limit_test_batches=1

 
