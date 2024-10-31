model_config = {
    "layers": 2,
    "hidden_channels": (16, 16),  # in each layer
    "temporal_kernel_size": (21, 11),
    "spatial_kernel_size": (11, 5),  # an int or tuple per layer
    "input_padding": False,
    "hidden_padding": True,
    "readout_positive": True,
    "readout_scale": True,
    "core_bias": True,
    "gaussian_masks": True,
    "nonlinearity": "ELU",
    "conv_type": "custom_separable",
    "stack": -1,  # from which layer (or list of layers) to readout responses from
    "gaussian_mean_scale": 6.0,  # from here on regularisation parameters
    "gaussian_var_scale": 4.0,
    "batch_adaptation": True,
    "gamma_readout": 0.4,
    "gamma_masks": 0.1,
    "gamma_input": 0.3,
    "gamma_in_sparse": 1.0,
    "gamma_hidden": 0.0,
    "gamma_temporal": 40.0,
}

trainer_config = {
    "max_iter": 500,
    "scale_loss": True,
    "lr_decay_steps": 4,
    "tolerance": 0.0005,
    "patience": 5,
    "verbose": False,
    "lr_init": 0.01,
    "avg_loss": False,
    "loss_function": "PoissonLoss3d",
    "stop_function": "corr_stop3d",
    "parallel": True,
}

STIMULUS_RANGE_CONSTRAINTS = {
    "norm": 30.0,
    "x_min_green": -0.654,
    "x_max_green": 6.269,
    "x_min_uv": -0.913,
    "x_max_uv": 6.269,
}

pre_normalisation_values_18x16 = {
    "channel_0_mean": 37.417128327480455,
    "channel_0_std": 28.904812895781816,
    "channel_1_mean": 36.13151406772875,
    "channel_1_std": 39.84109959857139,
}


MEAN_STD_DICT_74x64 = {
    'channel_0_mean': 37.19756061097937,
    'channel_0_std': 30.26892576088715,
    'channel_1_mean': 36.76101593081903,
    'channel_1_std': 42.65469417011324,
    'joint_mean': 36.979288270899204,
    'joint_std': 36.98463253226166,
}

