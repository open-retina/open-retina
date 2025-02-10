
test-types:
	mypy openretina/ tests/

test-codestyle:
	ruff check openretina/ tests/

test-formatting:
	ruff format --check --diff openretina/ tests/

test-unittests:
	pytest tests/

test-all: test-types test-codestyle test-formatting test-unittests

fix-codestyle:
	ruff check openretina/ tests/ --fix

fix-formatting:
	ruff format openretina/ tests/

fix-all: fix-codestyle fix-formatting

test-corereadout:
	openretina train --config-path configs --config-name hoefling_2024_core_readout_low_res \
	data.responses_path=https://gin.g-node.org/teulerlab/open-retina/raw/master/responses/eulerlab/rgc_natsim_subset_only_naturalspikes_2024-08-14.h5 \
	data.output_dir=exp exp_name=test_hoefling_2024_low_res \
	+trainer.limit_train_batches=1 trainer.max_epochs=1 +trainer.limit_val_batches=1 +trainer.limit_test_batches=1 dataloader.batch_size=2

test-notebooks:
	pytest --nbmake --ignore="notebooks/model_comparisons.ipynb" --ignore-glob="notebooks/training*" notebooks/

