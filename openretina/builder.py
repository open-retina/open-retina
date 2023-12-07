from functools import partial


def get_model(
    model_fn,
    model_config,
    dataloaders=None,
    seed=None,
    data_info=None,
):
    """
    Resolves `model_fn` and invokes the resolved function with `model_config` keyword arguments as well as the `dataloader` and `seed`.
    Note that the resolved `model_fn` is expected to accept the `dataloader` as the first positional argument and `seed` as a keyword argument.
    If you pass in `state_dict`, the resulting nn.Module instance will be loaded with the state_dict, using appropriate `strict` mode for loading.

    Args:
        model_fn: string name of the model builder function path to be resolved. Alternatively, you can pass in a callable object and no name resolution will be performed.
        model_config: a dictionary containing keyword arguments to be passed into the resolved `model_fn`
        dataloaders: (a dictionary of) dataloaders to be passed into the resolved `model_fn` as the first positional argument
        seed: randomization seed to be passed in to as a keyword argument into the resolved `model_fn`
        state_dict: If provided, the resulting nn.Module object will be loaded with the state_dict before being returned
        strict: Controls the `strict` mode of nn.Module.load_state_dict

    Returns:
        Resulting nn.Module object.
    """

    net = (
        model_fn(dataloaders, seed=seed, **model_config)
        if data_info is None
        else model_fn(dataloaders, data_info=data_info, seed=seed, **model_config)
    )

    return net


def get_data(dataset_fn, dataset_config):
    """
    Resolves `dataset_fn` and invokes the resolved function onto the `dataset_config` configuration dictionary. The resulting
    dataloader will be returned.

    Args:
        dataset_fn: string name of the dataloader function path to be resolved. Alternatively, you can pass in a callable object and no name resolution will be performed.
        dataset_config: a dictionary containing keyword arguments to be passed into the resolved `dataset_fn`

    Returns:
        Result of invoking the resolved `dataset_fn` with `dataset_config` as keyword arguments.
    """

    return dataset_fn(**dataset_config)


def get_trainer(trainer_fn, trainer_config=None):
    """
    If `trainer_fn` string is passed, resolves and returns the corresponding function. If `trainer_config` is passed in,
    a partial function is created with the configuration object expanded.

    Args:
        trainer_fn: string name of the function path to be resolved. Alternatively, you can pass in a callable object and no name resolution will be performed.
        trainer_config: If passed in, a partial function will be created expanding `trainer_config` as the keyword arguments into the resolved trainer_fn

    Returns:
        Resolved trainer function
    """

    if trainer_config is not None:
        trainer_fn = partial(trainer_fn, **trainer_config)

    return trainer_fn


def prepare_training(
    dataset_fn,
    dataset_config,
    model_fn,
    model_config,
    seed=None,
    state_dict=None,
    strict=True,
    trainer_fn=None,
    trainer_config=None,
):
    if seed is not None and "seed" not in dataset_config:
        dataset_config["seed"] = seed  # override the seed if passed in

    dataloaders = get_data(dataset_fn, dataset_config)

    model = get_model(
        model_fn,
        model_config,
        dataloaders,
        seed=seed,
        state_dict=state_dict,
        strict=strict,
    )

    if trainer_fn is not None:
        trainer = get_trainer(trainer_fn, trainer_config)
        return dataloaders, model, trainer
    else:
        return dataloaders, model
