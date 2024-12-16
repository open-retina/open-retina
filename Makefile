
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
	python ./scripts/train.py --config-path ../configs --config-name hoefling_2024_core_readout_low_res \
	data.root_dir=./openretina_cache_folder data.responses_path=https://gin.g-node.org/teulerlab/open-retina/raw/master/responses/eulerlab/rgc_natsim_subset_only_naturalspikes_2024-08-14.h5 \
	exp_name=test_hoefling_2024_low_res \
	+trainer.limit_train_batches=1 trainer.max_epochs=1 +trainer.limit_val_batches=1 +trainer.limit_test_batches=1 dataloader.batch_size=2

