
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
	data.responses_path=https://huggingface.co/datasets/open-retina/open-retina/blob/main/euler_lab/hoefling_2024/responses/rgc_natsim_subset_only_naturalspikes_2024-08-14.h5 \
	data.output_dir=exp exp_name=test_hoefling_2024_low_res \
	+trainer.limit_train_batches=1 trainer.max_epochs=1 +trainer.limit_val_batches=1 +trainer.limit_test_batches=1 dataloader.batch_size=2

test-h5train:
	openretina create-data ./test_data; \
	openretina train --config-path configs --config-name h5_core_readout \
	data.root_dir="." \
	+trainer.limit_train_batches=1 trainer.max_epochs=1 +trainer.limit_val_batches=1 +trainer.limit_test_batches=1 dataloader.batch_size=2

test-notebooks:  # ignore demo notebook and training notebooks as they only run fast on GPU
	pytest --nbmake --ignore-glob="notebooks/training*" notebooks/

