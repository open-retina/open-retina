from functools import partial
from typing import Dict, List, Literal, Optional, Tuple

import lightning
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float, Int
from torch import nn

from openretina.cell_type_training import BADEN_GROUPS_MAP, extract_readout_masks
from openretina.dataloaders import DataPointWithMeta
from openretina.measures import CorrelationLoss3d, PoissonLoss3d
from openretina.models.gru_core import ConvGRUCore
from openretina.utils.misc import tensors_to_device


class IndependentReadout3d(nn.Module):
    def __init__(
        self,
        core_channels=64,
        positive=False,
        nonlinearity=True,
        neurons_attention=True,
        time_attention=False,
        return_channels=False,
        neurons_attention_kwargs: Optional[dict] = None,
        time_attention_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        A readout layer with frozen factorised masks and (optional) self attention, that expects feature weights,
        cell classes and spatials masks as input in the forward pass.
        To be used in conjunction with a core that outputs feature weights through a shifter network.

        NB: This readout has no masks, which are expected to be passed in the forward pass.

        Args:
            core_channels (int): The number of channels in the core output. Defaults to 64.
            positive (bool, optional): Whether the output should be positive. Defaults to False.
            nonlinearity (bool, optional): Whether to include a final softplus nonlinearity. Defaults to True.
            neurons_attention (bool, optional): Whether to include self-attention over neurons. Defaults to True.
            time_attention (bool, optional): Whether to include self-attention over time. Defaults to False.
            return_channels (bool, optional): Whether to return the output over channels, before ELU. Defaults to False.
        """
        super().__init__()
        self.core_channels = core_channels
        self.positive = positive
        self.nonlinearity = nonlinearity
        self.neurons_attention = neurons_attention
        self.time_attention = time_attention
        self.return_channels = return_channels

        if self.neurons_attention:
            # self.embed_cell_classes = nn.Embedding(len(BADEN_GROUPS), c)
            if neurons_attention_kwargs is None:
                neurons_attention_kwargs = {"nhead": 2, "dim_feedforward": 32, "dropout": 0.1}
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.core_channels, **neurons_attention_kwargs)

        if self.time_attention:
            # For time dependent attention, we need to add positional encoding
            max_len = 2000
            d_model = self.core_channels
            # Create the positional encoding matrix
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)
            if time_attention_kwargs is None:
                time_attention_kwargs = {"nhead": 2, "dim_feedforward": 32, "dropout": 0.1}

            self.time_encoder_layer = nn.TransformerEncoderLayer(d_model=self.core_channels, **time_attention_kwargs)

    def initialize(self, *args, **kwargs):
        """
        Added for compatibility with neuralpredictors
        """
        pass

    def regularizer(self):
        """
        Added for compatibility with neuralpredictors
        """
        return 0

    def forward(
        self,
        x: Float[torch.Tensor, "batch time channels width height"],
        features: Float[torch.Tensor, "batch time channels neurons"],
        spatial_masks: Float[torch.Tensor, "width height neurons"],
        scale: Optional[Float[torch.Tensor, " neurons"]] = None,
        bias: Optional[Float[torch.Tensor, " neurons"]] = None,
        subs_idx=None,
    ) -> Float[torch.Tensor, "batch time neurons"] | Float[torch.Tensor, "batch time channels neurons"]:
        b, t, c, w, h = x.size()

        n_neurons = features.shape[-1]

        features = features.reshape(b * t, c, n_neurons)
        x = x.reshape(b * t, c, w, h)

        if self.positive:
            torch.clamp(features, min=0.0)

        if subs_idx is not None:
            feat = features[..., subs_idx]
            masks = spatial_masks[..., subs_idx]

        else:
            feat = features
            masks = spatial_masks

        y = torch.einsum("ncwh,whd->ncd", x, masks)
        # NB: not summing over features yet.
        y = y * feat

        if scale is not None:
            y = y * scale
        if bias is not None:
            y = y + bias if subs_idx is None else y + bias[subs_idx]

        if self.neurons_attention:
            # y is now (batch * time, channels, neurons), need to reshape to do attention on neurons.
            y = rearrange(y, "bt c n -> n bt c")

            # Compute self attention
            y = self.encoder_layer(y)

            # Reshape back to (batch*time, channels, neurons)
            y = rearrange(y, "n bt c -> bt c n")

        if self.time_attention:
            # Extract time and put neurons in batch
            y = rearrange(y, "(b t) c n -> t (b n) c", b=b)

            # Add positional encoding for time
            y = y + self.pe[:t, ...]

            y = self.time_encoder_layer(y)

            # Reshape back to (batch*time, channels, neurons) to preapare for output
            y = rearrange(y, "t (b n) c -> (b t) c n", b=b)

        if self.return_channels:
            y = y.view(b, t, c, n_neurons)
            return y

        # Sum over channels
        y = y.sum(1)

        if self.nonlinearity:
            y = F.softplus(y)

        y = y.view(b, t, n_neurons)

        return y

    def __repr__(self):
        return f"{self.__class__.__name__} (" + f"{self.core_channels} x w x h" + " -> " + "n_neurons" + ")"

    def save_weight_visualizations(self, folder_path: str) -> None:
        raise NotImplementedError("weight visualizations not implemented for this readout yet")


class ReadoutWeightShifter(nn.Module):
    def __init__(
        self,
        num_numerical_features,
        categorical_vocab_sizes,
        categorical_embedding_dims,
        output_dim,
        num_layers=2,
        hidden_units=(64, 32),
        use_bn=True,
        tanh_output=True,
        sigmoid_output=False,  # TODO pass class directly
        learn_scale=False,
        learn_bias=False,
        gamma_activations=0.1,
        gamma_variance=0.0,
    ):
        super(ReadoutWeightShifter, self).__init__()

        if sigmoid_output and tanh_output:
            raise ValueError("Only one of sigmoid_output and tanh_output can be True.")

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(vocab_size, categorical_embedding_dim)
                for vocab_size, categorical_embedding_dim in zip(
                    categorical_vocab_sizes, categorical_embedding_dims, strict=True
                )
            ]
        )

        # Dense layers for numerical features
        self.num_numerical_features = num_numerical_features
        self.fc1 = nn.Linear(
            num_numerical_features + int(np.sum(categorical_embedding_dims)),
            hidden_units[0],
        )
        self.bn1 = nn.BatchNorm1d(hidden_units[0])

        # Additional hidden layers
        self.hidden_layers = nn.ModuleList()
        for i, _ in enumerate(range(num_layers - 1)):
            self.hidden_layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            self.hidden_layers.append(nn.GELU())
            self.hidden_layers.append(nn.Dropout(0.1))

        # Output layer
        additional_output = (1 if learn_bias else 0) + (1 if learn_scale else 0)
        self.output_layer = nn.Linear(hidden_units[-1], output_dim + additional_output)

        # Additional parameters
        self.use_bn = use_bn
        self.learn_scale = learn_scale
        self.learn_bias = learn_bias

        self.final_nonlinearity = nn.Sigmoid() if sigmoid_output else nn.Tanh() if tanh_output else nn.Identity()

        # Regularization
        self.gamma_activations = gamma_activations
        self.gamma_variance = gamma_variance

    def forward(
        self,
        categorical_inputs: List[Float[torch.Tensor, "batch n_neurons"]],
        numerical_input: Float[torch.Tensor, "batch n_neurons n_features"],
    ) -> tuple[Float[torch.Tensor, "n_neurons n_features"], list]:
        # Batch dimensions is redundant, as all neuron come from the same session (with current loaders), so we remove
        # TODO: consider if this is the best way to handle this: can also reshape neurons and batch together
        categorical_inputs = [categorical_input[0] for categorical_input in categorical_inputs]
        numerical_input = numerical_input[0]

        # Embed the categorical inputs. Output is (n_neurons, n_cat_features)
        embedded_cats = [embedding(cat_input) for embedding, cat_input in zip(self.embeddings, categorical_inputs)]

        # Concatenate the embedded categorical features along the feature dimension
        all_embedded_cats = (
            torch.cat(embedded_cats, dim=-1) if embedded_cats else torch.tensor([], device=numerical_input.device)
        )

        # Concatenate numerical and embedded categorical features
        x = torch.cat([numerical_input, all_embedded_cats], dim=-1)

        x = self.fc1(x)

        x = F.gelu(self.bn1(x)) if self.use_bn else F.gelu(x)

        # Forward pass through the additional hidden layers
        for layer in self.hidden_layers:
            x = layer(x)

        x = self.final_nonlinearity(self.output_layer(x))

        return x, embedded_cats


class SessionAgnosticModel(lightning.LightningModule):
    """
    Represents a session-agnostic model for training and evaluation using PyTorch Lightning.

    Args:
        in_shape (tuple[int, int, int, int]): The shape of the input data (channels, time, height, width).
        in_channels (int): The number of input channels.
        hidden_channels (Tuple[int, ...]): The number of hidden channels for each layer.
        temporal_kernel_sizes (Tuple[int, ...]): The sizes of the temporal kernels for each layer.
        spatial_kernel_sizes (Tuple[int, ...]): The sizes of the spatial kernels for each layer.
        readout_mask_from (Optional[nn.Module]): A model from which to initialize readout masks.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        interval (Literal["epoch", "step"]): The interval for updating the learning rate.
        core_conditioning (bool, optional): Whether to use core conditioning. Defaults to False.
        shifter_num_numerical_features (int, optional): The number of numerical features for the shifter. Defaults to 5.
        shifter_categorical_vocab_sizes (Tuple[int, ...], optional): The vocabulary sizes for categorical features.
                                                                    Defaults to (47,).
        shifter_categorical_embedding_dims (Tuple[int, ...], optional): The embedding dimensions for
                                                                        categorical features. Defaults to (10,).
        shifter_num_layer (int, optional): The number of layers in the shifter. Defaults to 2.
        shifter_hidden_units (Tuple[int, ...], optional): The number of hidden units in the shifter layers.
                                                            Defaults to (64, 32).
        shifter_batch_norm (bool, optional): Whether to use batch normalization in the shifter. Defaults to True.
        shifter_tanh_output (bool, optional): Whether to use a tanh activation for the shifter output. Defaults to True.
        shifter_gamma (float, optional): The gamma parameter for the shifter. Defaults to 0.0.
        shifter_gamma_variance (float, optional): The variance of the gamma parameter for the shifter. Defaults to 0.0.
        readout_scale (bool, optional): Whether to learn a scale for the readout. Defaults to False.
        readout_bias (bool, optional): Whether to learn a bias for the readout. Defaults to False.
        readout_neurons_attention (bool, optional): Whether to use attention on neurons in the readout.
                                                    Defaults to True.
        readout_time_attention (bool, optional): Whether to use attention on time in the readout. Defaults to False.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        readout_neurons_attention_kwargs (Optional[dict], optional): Additional arguments for neuron attention.
                                                                    Defaults to None.
        readout_time_attention_kwargs (Optional[dict], optional): Additional arguments for time attention.
                                                                    Defaults to None.


    """

    def __init__(
        self,
        in_shape: Int[list | tuple, "channels time height width"],
        hidden_channels: Tuple[int, ...],
        temporal_kernel_sizes: Tuple[int, ...],
        spatial_kernel_sizes: Tuple[int, ...],
        readout_mask_from: Optional[nn.Module],
        optimizer: partial[torch.optim.Optimizer],
        scheduler: partial[torch.optim.lr_scheduler._LRScheduler],
        scheduler_interval: Literal["epoch", "step"],
        core_conditioning: bool = False,
        core_gamma_input: float = 0.3,
        core_gamma_in_sparse: float = 1.0,
        core_gamma_temporal: float = 40.0,
        core_gamma_hidden: float = 0.0,
        core_bias: bool = True,
        core_input_padding: bool = False,
        core_hidden_padding: bool = True,
        core_use_projections: bool = True,
        core_use_gru: bool = False,
        core_gru_kwargs: Optional[dict] = None,
        shifter_num_numerical_features: int = 5,
        shifter_categorical_vocab_sizes: Tuple[int, ...] = (47,),
        shifter_categorical_embedding_dims: Tuple[int, ...] = (10,),
        shifter_num_layer: int = 2,
        shifter_hidden_units: Tuple[int, ...] = (64, 32),
        shifter_batch_norm: bool = True,
        shifter_tanh_output: bool = True,
        shifter_gamma: float = 0.0,
        shifter_gamma_variance: float = 0.0,
        readout_positive: bool = False,
        readout_nonlinearity: bool = True,
        readout_scale: bool = False,
        readout_bias: bool = False,
        readout_neurons_attention: bool = True,
        readout_time_attention: bool = False,
        learning_rate: float = 0.01,
        readout_neurons_attention_kwargs: Optional[dict] = None,
        readout_time_attention_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer", "scheduler", "readout_mask_from"])

        if core_conditioning:
            raise NotImplementedError("Core conditioning is not implemented yet in the lightning interface.")

        self.core = ConvGRUCore(
            input_channels=in_shape[0],
            hidden_channels=hidden_channels,
            temporal_kernel_size=temporal_kernel_sizes,
            spatial_kernel_size=spatial_kernel_sizes,
            layers=len(hidden_channels),
            gamma_hidden=core_gamma_hidden,
            gamma_input=core_gamma_input,
            gamma_in_sparse=core_gamma_in_sparse,
            gamma_temporal=core_gamma_temporal,
            final_nonlinearity=True,
            bias=core_bias,
            input_padding=core_input_padding,
            hidden_padding=core_hidden_padding,
            batch_norm=True,
            batch_norm_scale=True,
            batch_norm_momentum=0.1,
            batch_adaptation=False,
            use_avg_reg=False,
            nonlinearity="ELU",
            conv_type="custom_separable",
            use_gru=core_use_gru,
            use_projections=core_use_projections,
            gru_kwargs=core_gru_kwargs,
        )

        self.core_conditioning = core_conditioning
        if self.core_conditioning:
            assert (
                len(shifter_categorical_embedding_dims) == 2
            ), "Two categorical features are required for conditioning - the second being the conditioning."

        # Run one forward path to determine output shape of core
        with torch.no_grad():
            core_test_output = self.core.forward(torch.zeros((1,) + tuple(in_shape)))

        self.readout = IndependentReadout3d(
            core_channels=hidden_channels[-1],
            positive=readout_positive,
            nonlinearity=readout_nonlinearity,
            neurons_attention=readout_neurons_attention,
            time_attention=readout_time_attention,
            return_channels=False,
            neurons_attention_kwargs=readout_neurons_attention_kwargs,
            time_attention_kwargs=readout_time_attention_kwargs,
        )

        self.readout_shifter = ReadoutWeightShifter(
            shifter_num_numerical_features,
            shifter_categorical_vocab_sizes,
            shifter_categorical_embedding_dims,
            output_dim=hidden_channels[-1],
            num_layers=shifter_num_layer,
            hidden_units=shifter_hidden_units,
            use_bn=shifter_batch_norm,
            tanh_output=shifter_tanh_output,
            learn_scale=readout_scale,
            learn_bias=readout_bias,
            gamma_activations=shifter_gamma,
            gamma_variance=shifter_gamma_variance,
        )

        # Initialise readout masks from another model
        if readout_mask_from is None:
            # TODO: In this case implement init of all masks to centered readout.
            raise NotImplementedError("readout_mask_from must be provided for now")

        masks = extract_readout_masks(
            readout_mask_from, ignore_source_key_suffix="_mb", target_dimensions=core_test_output.shape[-2:]
        )
        self.set_readout_mask_dict(masks)

        # Specify the train and evaluation loss
        self.loss = PoissonLoss3d()
        self.correlation_loss = CorrelationLoss3d(avg=True)
        self.learning_rate = learning_rate

        # Optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.interval = scheduler_interval

    def set_readout_mask_dict(self, readout_mask_dict: Dict[str, torch.Tensor]):
        self.readout_mask_dict = nn.ParameterDict()
        for name, tensor in readout_mask_dict.items():
            tensor.requires_grad = False
            # Register each tensor as a buffer
            self.register_buffer(f"{name}_readout_mask", tensor)
            # Also add it to readout_mask_dict for easy access
            self.readout_mask_dict[name] = tensor
        self.readout_mask_dict.requires_grad_(False)

    def add_readout_mask_dict(self, readout_mask_dict: Dict[str, torch.Tensor]):
        assert hasattr(self, "readout_mask_dict"), "No readout mask dict found. Use set_readout_mask_dict first."
        for name, tensor in readout_mask_dict.items():
            assert name not in self.readout_mask_dict, f"Readout mask for {name} already exists."
            assert tensor.shape == next(iter(self.readout_mask_dict.values())).shape, (
                f"Shape mismatch with existing readout masks. "
                f"{tensor.shape} != {next(iter(self.readout_mask_dict.values())).shape}"
            )
            tensor.requires_grad = False

            self.register_buffer(f"{name}_readout_mask", tensor)

            self.readout_mask_dict[name] = tensor

    @staticmethod
    def extract_baden_group(value):
        if value <= 9:
            group = "OFF"
        elif value >= 10 and value <= 14:
            group = "ON-OFF"
        elif value >= 15 and value <= 20:
            group = "fast ON"
        elif value >= 21 and value <= 28:
            group = "slow ON"
        elif value >= 28 and value <= 32:
            group = "uncertain RGC"
        else:
            group = "AC"
        return BADEN_GROUPS_MAP[group]

    @staticmethod
    def extract_baden_group_from_tensor(tensor):
        # Define the boundaries and corresponding groups for bucketization
        boundaries = torch.tensor([9, 14, 20, 28, 32], device=tensor.device)
        groups = torch.tensor(
            [
                BADEN_GROUPS_MAP["OFF"],
                BADEN_GROUPS_MAP["ON-OFF"],
                BADEN_GROUPS_MAP["fast ON"],
                BADEN_GROUPS_MAP["slow ON"],
                BADEN_GROUPS_MAP["uncertain RGC"],
                BADEN_GROUPS_MAP["AC"],
            ],
            device=tensor.device,
        )

        # Use bucketize to map cell types to group indices
        bucket_indices = torch.bucketize(tensor, boundaries)
        baden_groups = groups[bucket_indices]

        return baden_groups.to(tensor.device)

    def forward(
        self,
        x,
        categorical_metadata: List[torch.Tensor],
        numerical_metadata: torch.Tensor,
        data_key=None,
        detach_core=False,
        return_features=False,
        readout_masks=None,
        **kwargs,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        feature_weights, embedded_cats = self.readout_shifter(categorical_metadata, numerical_metadata)

        # Extract the scale and bias if they are learned
        if self.readout_shifter.learn_scale:
            readout_scale = feature_weights[..., -1]
            feature_weights = feature_weights[..., :-1]
        else:
            readout_scale = None

        if self.readout_shifter.learn_bias:
            readout_bias = feature_weights[..., -1]
            feature_weights = feature_weights[..., :-1]
        else:
            readout_bias = None

        self.detach_core = detach_core

        # We should not pass data specific information to the core, as it is session-independent,
        # unless conditioning is requested (i.e. the model version with session information)
        x = (
            self.core(
                x,
                repeat(
                    embedded_cats[1][0], "embed_d -> batch embed_d", batch=x.size(0)
                ),  # Session information assumed to be the second categorical feature. Extracting one as they repeat.
            )
            if self.core_conditioning
            else self.core(x)
        )
        if self.detach_core:
            x = x.detach()

        # Make time the second dimension for the readout
        x = torch.transpose(x, 1, 2)

        # Get dims for reshaping features
        batch_size = x.shape[0]
        time_points = x.shape[1]

        # Transpose the feature weights to have neurons as the last dimension
        feature_weights = feature_weights.T

        # Repeat the feature weights for each time point
        feature_weights = repeat(feature_weights, "d neurons -> b t d neurons", b=batch_size, t=time_points)

        # Even though the readout can be session-independent, it is going to be used
        # in a multiple-readout context during training, so we need to pass the data_key
        x = self.readout(
            x,
            feature_weights,
            spatial_masks=self.readout_mask_dict[data_key] if readout_masks is None else readout_masks,  # type: ignore
            scale=readout_scale,
            bias=readout_bias,
        )

        # Return the features if requested, used in regularisation
        return (x, feature_weights[0, 0, ...]) if return_features else x

    def compute_total_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, feature_weights: torch.Tensor, loss_scale: float = 1.0
    ) -> torch.Tensor:
        base_loss = self.loss.forward(predictions, targets)
        standard_regularizers = self.core.regularizer() + self.readout.regularizer()
        shifter_regularizers = (
            self.readout_shifter.gamma_activations * feature_weights.abs().sum()
            + self.readout_shifter.gamma_variance * (1.0 / (feature_weights.var(dim=0).mean() + 1e-5))
        )
        total_loss = loss_scale * base_loss + standard_regularizers + shifter_regularizers

        self.log("loss", base_loss)
        self.log("regularization_loss_core", self.core.regularizer())
        self.log("regularization_loss_readout", self.readout.regularizer())
        self.log("regularization_loss_shifter", shifter_regularizers)
        self.log("total_loss", total_loss)

        return total_loss

    ## Lightning methods below

    def training_step(
        self,
        batch: tuple[str, DataPointWithMeta],
        batch_idx: int,
        scale_loss: bool = False,
    ) -> torch.Tensor:
        session_id, data_point = batch
        predictions, feature_weights = self.forward(
            x=data_point.inputs,
            categorical_metadata=data_point.categorical_metadata,
            numerical_metadata=data_point.numerical_metadata,
            data_key=session_id,
            return_features=True,
        )

        if scale_loss:
            raise NotImplementedError("Loss scaling not implemented for this model yet")
            # m = self.dataset_length
            # # Assuming first input is always images, and batch size is the first dimension
            # k = data_point[0].shape[0]
            # loss_scale = np.sqrt(m / k)
        else:
            loss_scale = 1.0

        return self.compute_total_loss(predictions, data_point.targets, feature_weights, loss_scale)

    def validation_step(self, batch: tuple[str, DataPointWithMeta], batch_idx: int) -> torch.Tensor:
        session_id, data_point = batch
        model_output = self.forward(
            x=data_point.inputs,
            categorical_metadata=data_point.categorical_metadata,
            numerical_metadata=data_point.numerical_metadata,
            data_key=session_id,
            return_features=False,
        )

        loss = self.loss.forward(model_output, data_point.targets) / sum(model_output.shape)  # type: ignore
        correlation = -self.correlation_loss.forward(model_output, data_point.targets)
        self.log("val_loss", loss, logger=True, prog_bar=True)
        self.log("val_correlation", correlation, logger=True, prog_bar=True)

        return loss

    def test_step(self, batch: tuple[str, DataPointWithMeta], batch_idx: int, dataloader_idx) -> torch.Tensor:
        session_id, data_point = batch
        model_output = self.forward(
            x=data_point.inputs,
            categorical_metadata=data_point.categorical_metadata,
            numerical_metadata=data_point.numerical_metadata,
            data_key=session_id,
            return_features=False,
        )
        loss = self.loss.forward(model_output, data_point.targets) / sum(model_output.shape)  # type: ignore
        correlation = -self.correlation_loss.forward(model_output, data_point.targets)
        self.log_dict(
            {
                "test_loss": loss,
                "test_correlation": correlation,
            }
        )

        return loss

    def configure_optimizers(
        self,
    ):
        optimizer = self.optimizer(params=self.parameters())
        scheduler = self.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_correlation",
                "frequency": 1,
                "interval": self.interval,
            },
        }

    def save_weight_visualizations(self, folder_path: str) -> None:
        raise NotImplementedError("weight visualizations not implemented for this model yet")
