from abc import abstractmethod
import torch

class AbstractObjective:
    def __init__(self, model):
        self._model = model

    @abstractmethod
    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        pass


class SingleNeuronObjective(AbstractObjective):

    def __init__(self, model, neuron_idx: int):
        super.__init__(model)
        self._neuron_idx = neuron_idx

    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        responses = self._model.forward(stimulus)
        single_response = responses[self._neuron_idx]
        # average over time dimension
        single_score = torch.mean(single_response)
