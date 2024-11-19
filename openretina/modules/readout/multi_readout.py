import os

import torch

from .factorised_gaussian import SimpleSpatialXFeature3d


class MultiGaussianReadoutWrapper(torch.nn.ModuleDict):
    """
    Multiple Sessions version of the SimpleSpatialXFeature3d factorised gaussian readout.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        n_neurons_dict: dict[str, int],
        scale: bool,
        bias: bool,
        gaussian_masks: bool,
        gaussian_mean_scale: float,
        gaussian_var_scale: float,
        positive: bool,
        gamma_readout: float,
        gamma_masks: float = 0.0,
        readout_reg_avg: bool = False,
    ):
        super().__init__()
        for k in n_neurons_dict:  # iterate over sessions
            n_neurons = n_neurons_dict[k]
            assert len(in_shape) == 4
            self.add_module(
                k,
                SimpleSpatialXFeature3d(  # add a readout for each session
                    in_shape,
                    n_neurons,
                    gaussian_mean_scale=gaussian_mean_scale,
                    gaussian_var_scale=gaussian_var_scale,
                    positive=positive,
                    scale=scale,
                    bias=bias,
                ),
            )

        self.gamma_readout = gamma_readout
        self.gamma_masks = gamma_masks
        self.gaussian_masks = gaussian_masks
        self.readout_reg_avg = readout_reg_avg

    def forward(self, *args, data_key: str | None, **kwargs) -> torch.Tensor:
        if data_key is None:
            readout_responses = []
            for readout_key in self.readout_keys():
                resp = self[readout_key](*args, **kwargs)
                readout_responses.append(resp)
            response = torch.concatenate(readout_responses, dim=0)
        else:
            response = self[data_key](*args, **kwargs)
        return response

    def regularizer(self, data_key: str) -> torch.Tensor:
        feature_loss = self[data_key].feature_l1(average=self.readout_reg_avg) * self.gamma_readout
        mask_loss = self[data_key].mask_l1(average=self.readout_reg_avg) * self.gamma_masks
        return feature_loss + mask_loss

    def readout_keys(self) -> list[str]:
        return sorted(self._modules.keys())

    def save_weight_visualizations(self, folder_path: str) -> None:
        for key in self.readout_keys():
            readout_folder = os.path.join(folder_path, key)
            os.makedirs(readout_folder, exist_ok=True)
            self._modules[key].save_weight_visualizations(readout_folder)
