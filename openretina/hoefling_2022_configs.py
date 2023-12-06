model_config = dict(
    layers=2,  # number of layers
    hidden_channels=(
        16,
        16,
    ),  # number of channels in each hidden layer (implies length must be equal to # layers)
    temporal_kernel_size=(
        21,
        11,
    ),  # size of temporal kernels (in frames) in each hidden layer
    spatial_kernel_size=(
        11,
        5,
    ),  # size of spatial kernels (in pixels) in each hidden layer
    input_padding=False,
    hidden_padding=True,
    readout_positive=True,
    readout_scale=True,
    core_bias=True,
    gaussian_masks=True,
    stack=-1,
    gaussian_mean_scale=6.0,  # these are scaling and regularisation parameters
    gaussian_var_scale=4.0,
    batch_adaptation=True,
    gamma_readout=0.4,
    gamma_masks=0.1,
    gamma_input=0.3,
    gamma_in_sparse=1.0,
    gamma_hidden=0.0,
    gamma_temporal=40.0,
)

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
    "device": "cuda",
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
