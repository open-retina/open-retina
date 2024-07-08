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
    "use_readout_rnn": False,
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

model_fn = "nnfabrik_euler.models.models.SFB3d_core_SxF3d_readout"

dataset_config = {
    "experimenters": ("Szatko", "Szatko", "Szatko", "Szatko"),
    "dates": ("2021-09-29", "2021-09-29", "2021-09-30", "2021-09-30"),
    "exp_nums": (1, 2, 1, 2),
    "keys": ("ventral1", "ventral2", "ventral1", "ventral2"),
    "dataset_idxss": ([1, 2], [1, 2, 3, 4, 5], [1], [1, 2, 3]),
    "stim_id": 5,
    "stim_hash": "8c18928c21901a1a4af8b7458655a736",  # identifier to retrieve from DJ movie stim
    "detrend_param_set_id": 2,
    "quality_threshold_movie": 0,
    "quality_threshold_chirp": 0.35,
    "quality_threshold_ds": 0.6,
    "spikes": True,
    "chunk_size": 50,
    "batch_size": 32,
    "seed": 1000,
    "cell_types": (),
    "cell_type_crit": "exclude",
    "qi_link": "or",
}

postprocessing_config = {
    "norm": 30,
    "x_min_green": -0.654,
    "x_max_green": 6.269,
    "x_min_uv": -0.913,
    "x_max_uv": 6.269,
}


pre_normalisation_values = {
    "channel_0_mean": 37.417128327480455,
    "channel_0_std": 28.904812895781816,
    "channel_1_mean": 36.13151406772875,
    "channel_1_std": 39.84109959857139,
}
