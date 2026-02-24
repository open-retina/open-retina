import functools
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch

log = logging.getLogger(__name__)


class ResponseReducer(ABC):
    def __init__(self, axis: int = 0):
        self._axis = axis

    @abstractmethod
    def forward(self, responses: torch.Tensor) -> torch.Tensor:
        pass


class MeanReducer(ResponseReducer):
    def __init__(self, axis: int = 0):
        super().__init__(axis)

    def forward(self, responses: torch.Tensor) -> torch.Tensor:
        return torch.mean(responses, dim=self._axis)


class SliceMeanReducer(ResponseReducer):
    def __init__(self, axis: int, start: int, length: int):
        """ """
        super().__init__(axis)
        self.start = start
        self.length = length

    def forward(self, responses: torch.Tensor) -> torch.Tensor:
        response_length = responses.shape[self._axis]
        length = min(response_length - self.start, self.length)

        if length <= 0:
            length = min(self.length, response_length)
            log.warning(
                f"Response length too small ({response_length=}) for given start index ({self.start=})."
                f" Setting start=0 and {length=} in {self.__class__.__name__}."
            )
            narrowed_responses = torch.narrow(responses, self._axis, start=0, length=length)
        else:
            narrowed_responses = torch.narrow(responses, self._axis, self.start, length)
        return torch.mean(narrowed_responses, dim=self._axis)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._axis=} {self.start=}, {self.length=})"


class AbstractObjective(ABC):
    def __init__(self, model, data_key: str | None):
        self._model = model
        self._data_key = data_key

    def model_forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        if self._data_key is not None:
            responses = self._model(stimulus, data_key=self._data_key)
        else:
            responses = self._model(stimulus)
        # squeeze batch dimension
        return responses.squeeze(0)

    @abstractmethod
    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        pass


class IncreaseObjective(AbstractObjective):
    def __init__(self, model, neuron_indices: list[int] | int, data_key: str | None, response_reducer: ResponseReducer):
        super().__init__(model, data_key)
        self._neuron_indices = [neuron_indices] if isinstance(neuron_indices, int) else neuron_indices
        self._response_reducer = response_reducer

    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        responses = self.model_forward(stimulus)
        # responses.shape = (time, neuron)
        selected_responses = responses[:, self._neuron_indices]
        mean_response = selected_responses.mean(dim=-1)
        # average over time dimension
        single_score = self._response_reducer.forward(mean_response)
        return single_score


class _ModuleHook:
    """This structure is used to store the output of this module.
    This is achieved by the hook_fn as a forward hook which is called on every forward
    path of this module.
    We then store the output tensor which can then be used downstream, e.g. to optimize an MEI.
    Pass name to the initializer to get verbose output.
    """

    def __init__(self, module: torch.nn.Module, name: str | None = None):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.module_output_tensor: torch.Tensor | None = None
        self.name = name

    def hook_fn(self, module: torch.nn.Module, input_: tuple, output: torch.Tensor):
        assert self.module is module
        self.module_output_tensor = output
        if self.name is not None:
            print(f"Module hook {self.name} hooked up {type(self.module)=} {self.module_output_tensor.shape=}")

    def close(self) -> None:
        self.hook.remove()


class InnerNeuronVisualizationObjective(AbstractObjective):
    def __init__(self, model, data_key: str | None, response_reducer: ResponseReducer):
        super().__init__(model, data_key)
        self.features_dict = self.hook_model(model)
        self._response_reducer = response_reducer
        self.layer_name = ""
        self.channel_id = -1

    def set_layer_channel(self, layer: str, channel: int) -> None:
        if layer not in self.features_dict:
            raise ValueError(f"{layer=} not in features {self.features_dict.keys=}")
        self.layer_name = layer
        self.channel_id = channel

    @functools.lru_cache
    def get_output_shape_for_layer(self, layer_name: str, stimulus_shape: tuple[int, ...]) -> tuple[int, ...] | None:
        if layer_name not in self.features_dict:
            return None

        with torch.no_grad():
            model_device = next(self._model.parameters()).device
            stimulus = torch.rand(stimulus_shape, device=model_device, requires_grad=False)
            self.model_forward(stimulus)
            output_tensor = self.features_dict[layer_name].module_output_tensor
            if output_tensor is None:
                output_shape = None
            else:
                output_shape = output_tensor.shape
        return output_shape

    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        if len(self.layer_name) == 0 or self.channel_id < 0:
            raise ValueError("layer_name and channel_id not set. Please call `set_layer_channel`.")
        # The forward call is just used to update the model hooks
        self.model_forward(stimulus)
        module_hook = self.features_dict[self.layer_name]
        if module_hook.module_output_tensor is None:
            raise ValueError(f"Module hook output tensor is None for layer {self.layer_name}: {module_hook=}")
        output = module_hook.module_output_tensor[:, self.channel_id]
        y, x = output.shape[-2:]
        middle_neuron_trace_batch = output[..., y // 2, x // 2]
        middle_neuron_trace = middle_neuron_trace_batch.mean(axis=0)
        loss = self._response_reducer.forward(middle_neuron_trace)
        return loss

    @staticmethod
    def hook_model(model) -> OrderedDict:
        features = OrderedDict()

        # recursive hooking function
        def hook_layers(net, prefix: tuple[str, ...] = ()) -> None:
            if hasattr(net, "_modules"):
                for name, layer in net._modules.items():
                    if layer is not None:
                        new_prefix_tuple = prefix + (name,)
                        features_key = "_".join(new_prefix_tuple)
                        features[features_key] = _ModuleHook(layer)
                        hook_layers(layer, prefix=new_prefix_tuple)

        hook_layers(model)

        return features


class ContrastiveNeuronObjective(AbstractObjective):
    """Objective described in [Most discriminative stimuli for functional cell type clustering]
    (https://openreview.net/forum?id=9W6KaAcYlr)"""

    def __init__(
        self,
        model,
        on_cluster_idc: list[int],
        off_cluster_idc_list: list[list[int]],
        data_key: str | None,
        response_reducer: ResponseReducer,
        temperature: float = 1.6,
    ):
        super().__init__(model, data_key)
        self._on_cluster_idc = on_cluster_idc
        self._off_cluster_idc_list = off_cluster_idc_list
        self._response_reducer = response_reducer
        self._temperature = temperature

    @staticmethod
    def contrastive_objective(on_score: torch.Tensor, all_scores: torch.Tensor, temperature: float) -> torch.Tensor:
        t = temperature
        obj = (
            (1 / t) * on_score
            - torch.logsumexp((1 / t) * all_scores, dim=0)
            + torch.log(torch.tensor(all_scores.size(0)))
        )
        return obj

    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        responses = self.model_forward(stimulus)
        score_per_neuron = self._response_reducer.forward(responses)

        on_score = score_per_neuron[self._on_cluster_idc].mean()
        off_scores = [score_per_neuron[idc].mean() for idc in self._off_cluster_idc_list]
        obj = self.contrastive_objective(
            on_score,
            torch.stack([on_score] + off_scores),
            self._temperature,
        )
        return obj
