from abc import abstractmethod, ABC
import torch


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
        super().__init__(axis)
        self._start = start
        self._length = length

    def forward(self, responses: torch.Tensor) -> torch.Tensor:
        narrowed_responses = torch.narrow(responses, self._axis, self._start, self._length)
        return torch.mean(narrowed_responses, dim=self._axis)


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


class SingleNeuronObjective(AbstractObjective):

    def __init__(self, model, neuron_idx: int, data_key: str | None, response_reducer: ResponseReducer):
        super().__init__(model, data_key)
        self._neuron_idx = neuron_idx
        self._response_reducer = response_reducer

    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        responses = self.model_forward(stimulus)
        # responses.shape = (time, neuron)
        single_response = responses[:, self._neuron_idx]
        # average over time dimension
        single_score = self._response_reducer.forward(single_response)
        return single_score


class ContrastiveNeuronObjective(AbstractObjective):
    def __init__(
            self,
            model,
            neuron_idx: int,
            data_key: str | None,
            response_reducer: ResponseReducer,
            temperature: float = 1.6,
    ):
        super().__init__(model, data_key)
        self._neuron_idx = neuron_idx
        self._response_reducer = response_reducer
        self._temperature = temperature

    @staticmethod
    def contrastive_objective(on_score, all_scores, temperature: float):
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
        obj = self.contrastive_objective(
            score_per_neuron[self._neuron_idx],
            score_per_neuron,
            self._temperature,
        )
        return obj
