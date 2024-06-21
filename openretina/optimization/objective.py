from abc import abstractmethod, ABC
import torch


class ResponseReducer(ABC):
    def __init__(self, axis: int = 0):
        self._axis = 0

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
        super().__init__(axis)
        self._start = start
        self._length = length

    def forward(self, responses: torch.Tensor) -> torch.Tensor:
        narrowed_responses = torch.narrow(responses, self._axis, self._start, self._length)
        return torch.mean(narrowed_responses, dim=self._axis)


class AbstractObjective(ABC):
    def __init__(self, model):
        self._model = model

    @abstractmethod
    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        pass


class SingleNeuronObjective(AbstractObjective):

    def __init__(self, model, neuron_idx: int, data_key: str, response_reducer: ResponseReducer):
        super().__init__(model)
        self._neuron_idx = neuron_idx
        self._data_key = data_key
        self._response_reducer = response_reducer

    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        responses = self._model(stimulus, data_key=self._data_key)
        # responses.shape = (batch=1, time, neuron)
        responses = responses.squeeze(axis=0)
        single_response = responses[:, self._neuron_idx]
        # average over time dimension
        single_score = self._response_reducer.forward(single_response)
        return single_score
