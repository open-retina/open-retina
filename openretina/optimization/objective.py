from abc import abstractmethod
import torch

class AbstractObjective:
    def __init__(self, model):
        self._model = model

    @abstractmethod
    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        pass


class SingleNeuronObjective(AbstractObjective):

    def __init__(self, model, neuron_idx: int, data_key: str):
        super().__init__(model)
        self._neuron_idx = neuron_idx
        self._data_key = data_key

    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        responses = self._model.forward(stimulus, data_key=self._data_key)
        # responses.shape = (batch, time, neuron)
        single_response = responses[:, :, self._neuron_idx]
        # average over time dimension
        single_score = torch.mean(single_response)
        return single_score
