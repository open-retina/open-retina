install:
	uv sync

install-dev:
	uv sync --extra dev

test-types:
	uv run mypy openretina/ tests/

test-codestyle:
	uv run ruff check openretina/ tests/

test-formatting:
	uv run ruff format --check --diff openretina/ tests/

test-unittests:
	uv run pytest tests/

test-all: test-types test-codestyle test-formatting test-unittests

fix-codestyle:
	uv run ruff check openretina/ tests/ --fix

fix-formatting:
	uv run ruff format openretina/ tests/

fix-all: fix-codestyle fix-formatting

test-corereadout:
	uv run openretina train --config-path configs --config-name hoefling_2024_core_readout_low_res \
	paths.responses_path=https://huggingface.co/datasets/open-retina/open-retina/blob/main/euler_lab/hoefling_2024/responses/rgc_natsim_subset_only_naturalspikes_2024-08-14.h5 \
	paths.output_dir=exp exp_name=test_hoefling_2024_low_res \
	+trainer.limit_train_batches=1 trainer.max_epochs=1 +trainer.limit_val_batches=1 +trainer.limit_test_batches=1 dataloader.batch_size=2

test-lnp:
	uv run openretina train --config-path configs --config-name hoefling_2024_core_readout_low_res \
	paths.responses_path=https://huggingface.co/datasets/open-retina/open-retina/blob/main/euler_lab/hoefling_2024/responses/rgc_natsim_subset_only_naturalspikes_2024-08-14.h5 \
	paths.output_dir=exp exp_name=test_hoefling_2024_low_res_lnp \
	model=linear_nonlinear_poisson

test-h5train:
	uv run openretina create-data ./test_data --num-colors 3 --num-stimuli 4 --num-sessions 2; \
	uv run openretina train --config-path configs --config-name hdf5_core_readout \
	paths.cache_dir="." \
	+trainer.limit_train_batches=1 trainer.max_epochs=1 +trainer.limit_val_batches=1 +trainer.limit_test_batches=1 dataloader.batch_size=2

test-notebooks:  # ignore compute and disk space heavy notebooks
	uv run pytest --nbmake \
		--ignore-glob='notebooks/training*.ipynb' \
		--ignore-glob='notebooks/vector_field_analysis*.ipynb' \
		--ignore-glob='notebooks/new_datasets_guide*.ipynb' \
		notebooks/

# Fast tool runs without creating a full project environment - used in actions
uvx-test-codestyle:
	uvx ruff check openretina/ tests/

uvx-test-formatting:
	uvx ruff format --check --diff openretina/ tests/
