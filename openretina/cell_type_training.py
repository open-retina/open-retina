from collections import namedtuple
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from neuralpredictors.layers.readouts import MultiReadoutBase, Readout
from scipy.fftpack import fft
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataloaders import MovieAndMetadataDataSet, MovieSampler
from .dev_models import (  # ! TODO: move the model to dev models once done developing.
    GRUEnabledCore,
    get_dims_for_loader_dict,
    get_module_output,
    itemgetter,
    set_seed,
)
from .hoefling_2024.constants import CLIP_LENGTH, NUM_CLIPS, NUM_VAL_CLIPS
from .hoefling_2024.data_io import (
    MoviesDict,
    gen_start_indices,
    get_all_movie_combinations,
)
from .neuron_data_io import NeuronData, upsample_traces
from .utils.misc import tensors_to_device

DataPoint = namedtuple("DataPoint", ("inputs", "targets"))


BARCODE_MEANS = {
    "chirp_features": np.array([[0.19, 0.68, 2.27, 1.24, 7.38]]),
    "ds_index": np.array([0.23]),
    "os_index": np.array([0.36]),
    "temporal_nasal_pos_um": np.array([271.27]),
    "chirp_qi": np.array([0.44]),
    "ventral_dorsal_pos_um": np.array([-764.9]),
    "d_qi": np.array([0.75]),
    "roi_size_um2": np.array([81.13]),
    "pref_dir": np.array([0.31]),
}

BARCODES_STDEVS = {
    "chirp_features": np.array([[0.03, 0.07, 0.72, 0.44, 2.52]]),
    "ds_index": np.array([0.12]),
    "os_index": np.array([0.14]),
    "temporal_nasal_pos_um": np.array([623.39]),
    "chirp_qi": np.array([0.2]),
    "ventral_dorsal_pos_um": np.array([392.18]),
    "d_qi": np.array([0.15]),
    "roi_size_um2": np.array([40.74]),
    "pref_dir": np.array([1.54]),
}

# Stimuli repetitions in the hoefling_2024 dataset
CHIRP_REPEATS = 5
MB_REPEATS = 3

BADEN_GROUPS = ["fast ON", "slow ON", "OFF", "ON-OFF", "AC", "uncertain RGC"]
BADEN_GROUPS_MAP = {"fast ON": 0, "slow ON": 1, "OFF": 2, "ON-OFF": 3, "AC": 4, "uncertain RGC": 5}
INVERSE_BADEN_GROUPS_MAP = {v: k for k, v in BADEN_GROUPS_MAP.items()}


def transfer_readout_mask(
    source_model: nn.Module,
    target_model: nn.Module,
    ignore_source_key_suffix: Optional[str] = None,
    freeze_mask: bool = False,
    return_masks: bool = False,
) -> Union[nn.Module, Dict[str, torch.Tensor]]:
    """
    Given a trained source model and a target model, transfer the readout masks from the source to the target model.

    Args:
        source_model (nn.Module): The trained source model from which to transfer the readout masks.
        target_model (nn.Module): The target model to which the readout masks will be transferred.
        ignore_source_key_suffix (str, optional): Suffix of the source model key to ignore when transferring the masks.
                                            This is useful when the source model was trained on e.g. moving bars.
                                            Default is None.
        freeze_mask (bool, optional): Whether to freeze the transferred masks in the target model. Default is False.

    Note:
        - The source and target models should have the same architecture.
        - The source and target models should be initialized on the same dataloaders (or at least have the same sessions
            if trained on stimuli of different types like moving bars and natural movies).

    """
    if return_masks:
        masks = {}
    for name, param in source_model.named_parameters():
        if "readout" in name and "mask" in name:
            if ignore_source_key_suffix is not None:
                split_name = name.split(".")
                session_key = split_name[1]
                session_key = "".join(session_key.split(ignore_source_key_suffix)[0])
                split_name[1] = session_key
                new_name = ".".join(split_name)
            else:
                new_name = name
            # copy the mask to the target model
            try:
                if return_masks:
                    masks[session_key] = param
                else:
                    target_model.state_dict()[new_name].copy_(param)
            except KeyError:
                print(f"Could not find {new_name} in the target model.")
                continue

    # Freeze the mask parameters if requested
    if freeze_mask:
        if return_masks:
            for name in masks:
                masks[name].requires_grad = False
        else:
            for name, param in target_model.named_parameters():
                if "readout" in name and "mask" in name:
                    param.requires_grad = False
    return masks if return_masks else target_model


def calculate_chirp_features(chirp_trace: Float[np.ndarray, "n_neurons n_timepoints"]):
    #! TODO: This needs to be tested. Also, needs to be computed on chirp average trace.
    def spectral_centroid(signal):
        magnitudes = np.abs(fft(signal))[: len(signal) // 2]
        normalized_frequencies = np.linspace(0, 0.5, len(magnitudes))
        return np.sum(normalized_frequencies * magnitudes) / np.sum(magnitudes)

    def spectral_flatness(signal):
        magnitudes = np.abs(fft(signal))[: len(signal) // 2]
        geometric_mean = np.exp(np.mean(np.log(magnitudes + np.finfo(float).eps)))
        arithmetic_mean = np.mean(magnitudes)
        return geometric_mean / (arithmetic_mean + np.finfo(float).eps)

    spectral_centroid_value = np.apply_along_axis(spectral_centroid, 1, chirp_trace)
    spectral_flatness_value = np.apply_along_axis(spectral_flatness, 1, chirp_trace)
    mean_value = np.mean(chirp_trace, axis=1)
    std_value = np.std(chirp_trace, axis=1)
    range_value = np.ptp(chirp_trace, axis=1)

    return np.stack([spectral_centroid_value, spectral_flatness_value, mean_value, std_value, range_value])


def generate_cell_barcodes(
    responses_dict, normalize=True, include_field_id=False
) -> Dict[str, Dict[str, Float[np.ndarray, "n_neurons n_features"]]]:
    cell_barcodes = {}

    if include_field_id:
        field_encoder = OrdinalEncoder()
        field_encoder.fit(np.array(list(responses_dict.keys())).reshape(-1, 1))

    for field_id in responses_dict:
        field_barcode = {
            "cell_types": responses_dict[field_id]["group_assignment"].astype(str),
            "chirp_features": calculate_chirp_features(responses_dict[field_id]["chirp_preprocessed_traces"]).T,
            "roi_size_um2": responses_dict[field_id]["roi_size_um2"][:, None],
            "chirp_qi": responses_dict[field_id]["chirp_qi"][:, None],
            "d_qi": responses_dict[field_id]["d_qi"][:, None],
            "ds_index": responses_dict[field_id]["ds_index"][:, None],
            "os_index": responses_dict[field_id]["os_index"][:, None],
            "pref_dir": responses_dict[field_id]["pref_dir"][:, None],
            "temporal_nasal_pos_um": np.repeat(
                responses_dict[field_id]["temporal_nasal_pos_um"], len(responses_dict[field_id]["group_assignment"])
            )[:, None],
            "ventral_dorsal_pos_um": np.repeat(
                responses_dict[field_id]["ventral_dorsal_pos_um"], len(responses_dict[field_id]["group_assignment"])
            )[:, None],
        }

        if include_field_id:
            field_barcode["field_id"] = (
                field_encoder.transform(np.repeat(field_id, len(responses_dict[field_id]["group_assignment"]))[:, None])
                .squeeze()
                .astype(int)
                .astype(str)
            )

        # Normalize the features before concatenating
        if normalize:
            for key in field_barcode:
                if key in BARCODE_MEANS:
                    field_barcode[key] = (field_barcode[key] - BARCODE_MEANS[key]) / BARCODES_STDEVS[key]

        cell_barcodes[field_id] = field_barcode

    return cell_barcodes


def extract_chirp_mb(response_dict) -> Dict[str, Dict[str, Float[np.ndarray, "n_neurons n_features "]]]:
    """
    Extracts and averages over repetitions chirp and moving bar (mb) traces from the response dictionary.

    Args:
        response_dict: A dictionary containing response traces for chirp and mb stimuli.

    Returns:
        dict: A dictionary with averaged chirp and mb traces for each field ID.
    """
    traces = {}

    for field_id in response_dict:
        field_average_traces = {
            "chirp": np.stack(
                np.split(
                    upsample_traces(
                        response_dict[field_id]["chirp_trigger_times"][0],
                        response_dict[field_id]["chirp_preprocessed_traces"],
                        response_dict[field_id]["chirp_traces_times"],
                        stim_id=1,
                    ),
                    CHIRP_REPEATS,
                    axis=1,
                ),
                axis=2,
            ).mean(2),
            "mb": np.stack(
                np.split(
                    upsample_traces(
                        response_dict[field_id]["mb_trigger_times"][0],
                        response_dict[field_id]["mb_preprocessed_traces"],
                        response_dict[field_id]["mb_traces_times"],
                        stim_id=2,
                    ),
                    MB_REPEATS,
                    axis=1,
                ),
                axis=2,
            ).mean(2),
        }

        traces[field_id] = field_average_traces

    return traces


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
    ) -> Float[torch.Tensor, "n_neurons n_features"]:
        # Batch dimensions is redundant, as all neuron come from the same session, so we remove it
        # TODO: consider if this is the best way to handle this: can also reshape neurons and batch together
        categorical_inputs = [categorical_input[0] for categorical_input in categorical_inputs]
        numerical_input = numerical_input[0]

        # Embed the categorical inputs. Output is (n_neurons, n_cat_features)
        embedded_cats = [embedding(cat_input) for embedding, cat_input in zip(self.embeddings, categorical_inputs)]

        # Concatenate the embedded categorical features along the feature dimension
        embedded_cats = (
            torch.cat(embedded_cats, dim=-1) if embedded_cats else torch.tensor([], device=numerical_input.device)
        )

        # Concatenate numerical and embedded categorical features
        x = torch.cat([numerical_input, embedded_cats], dim=-1)

        x = self.fc1(x)

        x = F.gelu(self.bn1(x)) if self.use_bn else F.gelu(x)

        # Forward pass through the additional hidden layers
        for layer in self.hidden_layers:
            x = layer(x)

        return self.final_nonlinearity(self.output_layer(x))


class FrozenFactorisedReadout2d(Readout):
    def __init__(
        self,
        in_shape,
        outdims,
        positive=False,
        nonlinearity=True,
        attention=True,
        return_channels=False,
        attention_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        A readout layer with frozen factorised masks and (optional) self attention, that expects feature weights,
        cell classes and spatials masks as input in the forward pass.
        To be used in conjunction with a core that outputs feature weights through a shifter network.

        NB: This readout has no masks, which are expected to be passed in the forward pass.

        Args:
            in_shape (tuple): The shape of the input tensor (c, t, w, h).
            outdims (int): The number of output dimensions (usually the number of neurons in the session).
            from_gaussian (bool, optional): Whether the masks are coming from a readout with Gaussian masks.
                                            Defaults to False.
            positive (bool, optional): Whether the output should be positive. Defaults to False.
            scale (bool, optional): Whether to include a scale parameter. Defaults to False.
            nonlinearity (bool, optional): Whether to include a nonlinearity. Defaults to True.
        """
        super().__init__()
        self.in_shape = in_shape
        c, w, h = in_shape
        self.outdims = outdims
        self.positive = positive
        self.nonlinearity = nonlinearity
        self.attention = attention
        self.return_channels = return_channels

        if self.attention:
            self.embed_cell_classes = nn.Embedding(len(BADEN_GROUPS), c)
            if attention_kwargs is None:
                attention_kwargs = {"nhead": 2, "dim_feedforward": 32, "dropout": 0.1}
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=c, **attention_kwargs)

    def initialize(self, *args, **kwargs):
        """
        Added for compatibility with neuralpredictors
        """
        pass

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels width height"],
        features: Float[torch.Tensor, "batch channels neurons"],
        spatial_masks: Float[torch.Tensor, "width height neurons"],
        cell_classes: Float[torch.Tensor, "batch neurons"],
        scale: Optional[Float[torch.Tensor, " neurons"]] = None,
        bias: Optional[Float[torch.Tensor, " neurons"]] = None,
        subs_idx=None,
    ):
        b, c, w, h = x.size()

        assert features.shape[-1] == self.outdims, "Number of neurons in features does not match outdims"

        features = features.view(b, c, self.outdims)

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

        if self.attention:
            # Compute embedding for neuron class, and add to the output
            embed_cell_classes = self.embed_cell_classes(cell_classes)

            # If batch dim includes time, need to expand the embedding to match the output
            if embed_cell_classes.size(0) != b:
                time = y.size(0) // embed_cell_classes.size(0)
                neurons = y.size(-1)
                embed_cell_classes = embed_cell_classes.unsqueeze(1).expand(-1, time, -1, -1).reshape(-1, neurons, c)

            y = y + embed_cell_classes.transpose(1, 2)

            # y is now (batch, channels, neurons), need to reshape to do attention on neurons.
            y = y.permute(2, 0, 1)

            # Compute self attention
            y = self.encoder_layer(y)

            # Reshape back to (batch, channels, neurons)
            y = y.permute(1, 2, 0)

        if self.return_channels:
            return y

        # Sum over channels
        y = y.sum(1)

        if self.nonlinearity:
            y = F.softplus(y)

        return y

    def __repr__(self):
        c, h, w = self.in_shape
        r = f"{self.__class__.__name__} (" + f"{c} x {w} x {h}" + " -> " + str(self.outdims) + ")"
        return


class MultipleFrozenFactorisedReadout2d(MultiReadoutBase):
    def __init__(
        self,
        in_shape_dict,
        n_neurons_dict,
        **readout_kwargs,
    ):
        super().__init__(
            in_shape_dict,
            n_neurons_dict,
            base_readout=FrozenFactorisedReadout2d,
            **readout_kwargs,
        )
        for kwarg in readout_kwargs:
            setattr(self, kwarg, readout_kwargs[kwarg])

    def regularizer(self, data_key):
        return 0


class ShifterVideoEncoder(nn.Module):
    """
    Video Encoder model that uses a session-independent core and a session-dependent readout with a weight shifter,
    to predict responses to a movie stimulus using cell "barcodes".
    """

    def __init__(
        self,
        core,
        readout,
        readout_shifter: ReadoutWeightShifter,
        readout_mask_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.core = core
        self.readout = readout
        self.readout_shifter = readout_shifter
        self.detach_core = False

        if readout_mask_dict is not None:
            self.set_readout_mask_dict(readout_mask_dict)

    def set_readout_mask_dict(self, readout_mask_dict: Dict[str, torch.Tensor]):
        self.readout_mask_dict = nn.ParameterDict()
        for name, tensor in readout_mask_dict.items():
            # Register each tensor as a buffer
            self.register_buffer(f"{name}_readout_mask", tensor)
            # Also add it to readout_mask_dict for easy access
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
        **kwargs,
    ):
        self.detach_core = detach_core
        # We should not pass data specific information to the core.
        x = self.core(x)
        if self.detach_core:
            x = x.detach()

        feature_weights = self.readout_shifter(categorical_metadata, numerical_metadata)

        # Cell types are assumed to be always the first passed categorical metadata
        cell_types = categorical_metadata[0]

        baden_groups = self.extract_baden_group_from_tensor(cell_types)

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

        # Make time the second dimension again for the readout
        x = torch.transpose(x, 1, 2)

        # Get dims for later reshaping
        batch_size = x.shape[0]
        time_points = x.shape[1]

        # Transpose the feature weights to have neurons as the last dimension
        feature_weights = feature_weights.T

        # Repeat the feature weights for each time point
        feature_weights = feature_weights.unsqueeze(0).unsqueeze(0).repeat(batch_size, time_points, 1, 1)

        # Treat time as an indipendent (batch) dimension for the readout
        x = x.reshape(((-1,) + x.size()[2:]))

        # Even though the readout can be session-independent, it is going to be used
        # in a multiple-readout context during training, so we need to pass the data_key
        x = self.readout(
            x,
            feature_weights,
            spatial_masks=self.readout_mask_dict[data_key],  # type: ignore
            cell_classes=baden_groups,
            scale=readout_scale,
            bias=readout_bias,
            data_key=data_key,
        )

        # Reshape back to the correct dimensions before returning
        x = x.reshape(((batch_size, time_points) + x.size()[1:]))

        # Return the features if requested, used in regularisation
        return (x, feature_weights[0, 0, ...]) if return_features else x


def conv_core_frozen_readout(
    dataloaders,
    seed,
    readout_mask_from: nn.Module,
    hidden_channels: Tuple[int, ...] = (8,),  # core args
    temporal_kernel_size: Tuple[int, ...] = (21,),
    spatial_kernel_size: Tuple[int, ...] = (11,),
    layers: int = 1,
    gamma_hidden: float = 0,
    gamma_input: float = 0.1,
    gamma_temporal: float = 0.1,
    gamma_in_sparse=0.0,
    final_nonlinearity: bool = True,
    core_bias: bool = False,
    momentum: float = 0.1,
    input_padding: bool = False,
    hidden_padding: bool = True,
    batch_norm: bool = True,
    batch_norm_scale: bool = False,
    laplace_padding: Optional[int] = None,
    readout_scale: bool = False,
    readout_bias: bool = False,
    readout_attention: bool = False,
    shifter_num_numerical_features: int = 5,
    shifter_categorical_vocab_sizes: Tuple[int, ...] = (47,),
    shifter_categorical_embedding_dims: Tuple[int, ...] = (10,),
    shifter_num_layer: int = 2,
    shifter_hidden_units: Tuple[int, ...] = (64, 32),
    shifter_batch_norm: bool = True,
    shifter_tanh_output: bool = True,
    shifter_gamma: float = 0.0,
    shifter_gamma_variance: float = 0.0,
    stack=None,
    use_avg_reg: bool = False,
    data_info: Optional[dict] = None,
    nonlinearity: str = "ELU",
    conv_type: Literal["full", "separable", "custom_separable", "time_independent"] = "custom_separable",
    use_gru: bool = False,
    use_projections: bool = False,
    gru_kwargs: dict = {},
    readout_attn_kwargs: dict = {},
    **kwargs,
):
    """
    TODO docstring
    """

    # make sure trainloader is being used
    if data_info is not None:
        in_shapes_dict = {k: v["input_dimensions"] for k, v in data_info.items()}
        input_channels = [v["input_channels"] for k, v in data_info.items()]
        n_neurons_dict = {k: v["output_dimension"] for k, v in data_info.items()}
    else:
        dataloaders = dataloaders.get("train", dataloaders)

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, *_, out_name = next(iter(list(dataloaders.values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders)

        n_neurons_dict = {
            k: v[out_name][-1] for k, v in session_shape_dict.items()
        }  # dictionary containing # neurons per session
        in_shapes_dict = {
            k: v[in_name] for k, v in session_shape_dict.items()
        }  # dictionary containing input shapes per session
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]  # gets the # of input channels
    assert np.unique(input_channels).size == 1, "all input channels must be of equal size"

    set_seed(seed)

    # get a stacked factorized 3d core from below
    core = GRUEnabledCore(
        n_neurons_dict=n_neurons_dict,
        input_channels=input_channels[0],
        num_scans=len(n_neurons_dict.keys()),
        hidden_channels=hidden_channels,
        temporal_kernel_size=temporal_kernel_size,
        spatial_kernel_size=spatial_kernel_size,
        layers=layers,
        gamma_hidden=gamma_hidden,
        gamma_input=gamma_input,
        gamma_in_sparse=gamma_in_sparse,
        gamma_temporal=gamma_temporal,
        final_nonlinearity=final_nonlinearity,
        bias=core_bias,
        momentum=momentum,
        input_padding=input_padding,
        hidden_padding=hidden_padding,
        batch_norm=batch_norm,
        batch_norm_scale=batch_norm_scale,
        laplace_padding=laplace_padding,
        stack=stack,
        batch_adaptation=False,
        use_avg_reg=use_avg_reg,
        nonlinearity=nonlinearity,
        conv_type=conv_type,
        use_gru=use_gru,
        use_projections=use_projections,
        gru_kwargs=gru_kwargs,
    )

    subselect = itemgetter(0, 2, 3)
    in_shapes_readout = {k: subselect(tuple(get_module_output(core, in_shapes_dict[k])[1:])) for k in n_neurons_dict}
    readout = MultipleFrozenFactorisedReadout2d(
        in_shape_dict=in_shapes_readout,
        n_neurons_dict=n_neurons_dict,
        nonlinearity=True,
        attention=readout_attention,
        return_channels=False,
        attention_kwargs=readout_attn_kwargs,
    )

    readout_shifter = ReadoutWeightShifter(
        shifter_num_numerical_features,
        shifter_categorical_vocab_sizes,
        shifter_categorical_embedding_dims,
        output_dim=core.hidden_channels[-1],
        num_layers=shifter_num_layer,
        hidden_units=shifter_hidden_units,
        use_bn=shifter_batch_norm,
        tanh_output=shifter_tanh_output,
        learn_scale=readout_scale,
        learn_bias=readout_bias,
        gamma_activations=shifter_gamma,
        gamma_variance=shifter_gamma_variance,
    )

    model = ShifterVideoEncoder(core, readout, readout_shifter)

    masks = transfer_readout_mask(
        readout_mask_from, model, ignore_source_key_suffix="_mb", freeze_mask=True, return_masks=True
    )

    model.set_readout_mask_dict(masks)

    return model


class NeuronDataWithBarcodes(NeuronData):
    def __init__(
        self,
        responses_final: Float[np.ndarray, "n_neurons n_timepoints"] | dict,
        stim_id: Literal[5, 2, 1, "salamander_natural"],
        val_clip_idx: Optional[List[int]],
        num_clips: Optional[int],
        clip_length: Optional[int],
        roi_ids: Optional[Float[np.ndarray, " n_neurons"]] = None,
        traces: Optional[Float[np.ndarray, "n_neurons n_timepoints"]] = None,
        tracestimes: Optional[Float[np.ndarray, " n_timepoints"]] = None,
        scan_sequence_idx: Optional[int] = None,
        random_sequences: Optional[Float[np.ndarray, "n_clips n_sequences"]] = None,
        eye: Optional[Literal["left", "right"]] = None,
        group_assignment: Optional[Float[np.ndarray, " n_neurons"]] = None,
        key: Optional[dict] = None,
        use_base_sequence: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            responses_final=responses_final,
            stim_id=stim_id,
            val_clip_idx=val_clip_idx,
            num_clips=num_clips,
            clip_length=clip_length,
            roi_ids=roi_ids,
            traces=traces,
            tracestimes=tracestimes,
            scan_sequence_idx=scan_sequence_idx,
            random_sequences=random_sequences,
            eye=eye,
            group_assignment=group_assignment,
            key=key,
            use_base_sequence=use_base_sequence,
            **kwargs,
        )


def get_movie_meta_dataloader(
    movies: np.ndarray | torch.Tensor | dict[int, np.ndarray],
    responses: Float[np.ndarray, "n_frames n_neurons"],
    metadata: Dict[str, Float[np.ndarray, "n_features n_neurons"]],
    split: str,
    start_indices: List[int] | Dict[int, List[int]],
    scan_sequence_idx: Optional[int] = None,
    chunk_size: int = 50,
    batch_size: int = 32,
    scene_length: Optional[int] = None,
    drop_last=True,
    use_base_sequence=False,
    **kwargs,
):
    """
    TODO docstring
    """
    if isinstance(responses, torch.Tensor) and bool(torch.isnan(responses).any()):
        print("Nans in responses, skipping this dataloader")
        return None

    # for right movie: flip second frame size axis!
    if split == "train" and isinstance(movies, dict) and scan_sequence_idx is not None:
        if use_base_sequence:
            scan_sequence_idx = 20  # 20 is the base sequence
        dataset = MovieAndMetadataDataSet(movies[scan_sequence_idx], responses, metadata, split, chunk_size)
        sampler = MovieSampler(
            start_indices[scan_sequence_idx],
            split,
            chunk_size,
            movie_length=movies[scan_sequence_idx].shape[1],
            scene_length=scene_length,
        )
    else:
        dataset = MovieAndMetadataDataSet(movies, responses, metadata, split, chunk_size)
        sampler = MovieSampler(
            start_indices, split, chunk_size, movie_length=movies.shape[1], scene_length=scene_length
        )

    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=split == "train" and drop_last,
        **kwargs,
    )


def natmov_and_meta_dataloaders(
    neuron_data_dictionary,
    movies_dictionary: MoviesDict,
    train_chunk_size: int = 50,
    batch_size: int = 32,
    seed: int = 42,
    num_clips: int = NUM_CLIPS,
    clip_length: int = CLIP_LENGTH,
    num_val_clips: int = NUM_VAL_CLIPS,
    use_base_sequence: bool = False,
    use_raw_traces: bool = False,
    include_field_info: bool = False,
):
    """
    Train, test and validation dataloaders for natural movies responses datasets and metadata.
    """
    assert isinstance(
        neuron_data_dictionary, dict
    ), "neuron_data_dictionary should be a dictionary of sessions and their corresponding neuron data."
    assert (
        isinstance(movies_dictionary, dict) and "train" in movies_dictionary and "test" in movies_dictionary
    ), "movies_dictionary should be a dictionary with keys 'train' and 'test'."
    assert all(field in next(iter(neuron_data_dictionary.values())) for field in ["responses_final", "stim_id"]), (
        "Check the neuron data dictionary sub-dictionaries for the minimal"
        " required fields: 'responses_final' and 'stim_id'."
    )

    assert next(iter(neuron_data_dictionary.values()))["stim_id"] in [
        5,
        "salamander_natural",
    ], "This function only supports natural movie stimuli."

    # Draw validation clips based on the random seed
    rnd = np.random.RandomState(seed)
    val_clip_idx = list(rnd.choice(num_clips, num_val_clips, replace=False))

    clip_chunk_sizes = {
        "train": train_chunk_size,
        "validation": clip_length,
        "test": movies_dictionary["test"].shape[1],
    }
    dataloaders: dict[str, Any] = {"train": {}, "validation": {}, "test": {}}

    # Get the random sequences of movies presentations for each session if available
    if "random_sequences" not in movies_dictionary or movies_dictionary["random_sequences"] is None:
        movie_length = movies_dictionary["train"].shape[1]
        random_sequences = np.arange(0, movie_length // clip_length)[:, np.newaxis]
    else:
        random_sequences = movies_dictionary["random_sequences"]
        if use_base_sequence:
            base_sequence = np.arange(num_clips)[:, None]
            random_sequences = np.concatenate([random_sequences, base_sequence], axis=1)

    movies = get_all_movie_combinations(
        movies_dictionary["train"],
        movies_dictionary["test"],
        random_sequences,
        val_clip_idx=val_clip_idx,
        clip_length=clip_length,
    )
    start_indices = gen_start_indices(random_sequences, val_clip_idx, clip_length, train_chunk_size, num_clips)

    # Extract cell barcodes from the neuron data dictionary
    if use_raw_traces:
        barcodes = extract_chirp_mb(neuron_data_dictionary)
    else:
        barcodes = generate_cell_barcodes(neuron_data_dictionary, include_field_id=include_field_info)

    for session_key, session_data in tqdm(neuron_data_dictionary.items(), desc="Creating movie dataloaders"):
        neuron_data = NeuronData(
            **session_data,
            random_sequences=random_sequences,
            val_clip_idx=val_clip_idx,
            num_clips=num_clips,
            clip_length=clip_length,
            use_base_sequence=use_base_sequence,
        )

        session_metadata = barcodes[session_key]

        for fold in ["train", "validation", "test"]:
            dataloaders[fold][session_key] = get_movie_meta_dataloader(
                movies=movies[neuron_data.eye][fold],
                responses=neuron_data.response_dict[fold],
                metadata=session_metadata,
                scan_sequence_idx=neuron_data.scan_sequence_idx,
                split=fold,
                chunk_size=clip_chunk_sizes[fold],
                start_indices=start_indices[fold],
                batch_size=batch_size,
                scene_length=clip_length,
                use_base_sequence=use_base_sequence,
            )

    return dataloaders


def metadata_model_full_objective(
    model: ShifterVideoEncoder,
    *inputs: torch.Tensor,
    targets,
    data_key,
    detach_core,
    device,
    criterion,
    scale_loss,
    dataset_length,
) -> torch.Tensor:
    """
    Slightly modified version of the standard training objective which includes regularisation on the readout `feature`
    weights, which in this specific model instance are activations of the readout shifter network and not parameters of
    the readout itself.
    """
    standard_regularizers = int(not detach_core) * model.core.regularizer() + model.readout.regularizer(data_key)
    if scale_loss:
        m = dataset_length
        # Assuming first input is always images, and batch size is the first dimension
        k = inputs[0].shape[0]
        loss_scale = np.sqrt(m / k)
    else:
        loss_scale = 1.0

    predictions, feature_weights = model(
        *tensors_to_device(inputs, device),
        data_key=data_key,
        detach_core=detach_core,
        return_features=True,
    )

    loss_criterion = criterion(predictions, targets.to(device))

    # Penalties encourage the feature weights to be sparse and to have high variance across neurons
    return (
        loss_scale * loss_criterion
        + standard_regularizers
        + model.readout_shifter.gamma_activations * feature_weights.abs().sum()
        + model.readout_shifter.gamma_variance * (1.0 / (feature_weights.var(dim=0).mean() + 1e-5))
    )
