import torch
from torch import nn


class CorrelationLoss3d(nn.Module):
    def __init__(self, bias: float = 1e-16, per_neuron: bool = False, avg: bool = False, negate: bool = True):
        super().__init__()
        self.eps = bias
        self.per_neuron = per_neuron
        self.avg = avg
        self.negate = negate

        # Placeholder to store last-computed per-neuron correlations
        self.register_buffer("_per_neuron_correlations", torch.tensor([]), persistent=False)

    def forward(self, output, target):
        lag = target.size(1) - output.size(1)
        delta_out = output - output.mean(1, keepdim=True)
        delta_target = target[:, lag:, :] - target[:, lag:, :].mean(1, keepdim=True)
        var_out = delta_out.pow(2).mean(1, keepdim=True)
        var_target = delta_target.pow(2).mean(1, keepdim=True)
        corrs = (delta_out * delta_target).mean(1, keepdim=True) / (
            (var_out + self.eps) * (var_target + self.eps)
        ).sqrt()

        per_neuron_correlations = corrs.view(-1, corrs.shape[-1]).mean(dim=0)
        self._per_neuron_correlations = per_neuron_correlations.detach()  # Not a loss, so no need to negate

        if self.negate:
            corrs = -corrs

        # Return scalar for possible backprop
        if not self.per_neuron:
            return corrs.mean() if self.avg else corrs.sum()
        else:
            return corrs.view(-1, corrs.shape[-1]).mean(dim=0)


class CelltypeCorrelationLoss3d(nn.Module):
    def __init__(self, bias: float = 1e-16, per_neuron: bool = False, avg: bool = False, negate: bool = True):
        super().__init__()
        self.eps = bias
        self.per_neuron = per_neuron
        self.avg = avg
        self.negate = negate
        # Placeholder to store last-computed per-neuron correlations
        self.register_buffer("_per_neuron_correlations", torch.tensor([]), persistent=False)

    def forward(self, output, target, group_assignment, group_counts):
        lag = target.size(1) - output.size(1)
        delta_out = output - output.mean(1, keepdim=True)
        delta_target = target[:, lag:, :] - target[:, lag:, :].mean(1, keepdim=True)
        var_out = delta_out.pow(2).mean(1, keepdim=True)
        var_target = delta_target.pow(2).mean(1, keepdim=True)
        corrs = (delta_out * delta_target).mean(1, keepdim=True) / (
            (var_out + self.eps) * (var_target + self.eps)
        ).sqrt()
        corrs = corrs * (group_counts.sum() / group_counts.size(0) / group_counts[group_assignment - 1])[:, None, ...]

        per_neuron_correlations = -corrs.view(-1, corrs.shape[-1]).mean(dim=0)
        self._per_neuron_correlations = per_neuron_correlations.detach()
        if self.negate:
            corrs = -corrs

        # Return scalar for backprop
        if not self.per_neuron:
            return corrs.mean() if self.avg else corrs.sum()
        else:
            return corrs.view(-1, corrs.shape[-1]).mean(dim=0)


class L1CorrelationLoss3d(nn.Module):
    def __init__(self, bias: float = 1e-16, per_neuron: bool = False, avg: bool = False, negate: bool = True):
        super().__init__()
        self.eps = bias
        self.per_neuron = per_neuron
        self.avg = avg
        self.gamma_L1 = 0.0002
        self.negate = negate
        # Placeholder to store last-computed per-neuron correlations
        self.register_buffer("_per_neuron_correlations", torch.tensor([]), persistent=False)

    def forward(self, output, target, **kwargs):
        pre_ca = output[1]
        output = output[0]
        lag = target.size(1) - output.size(1)
        delta_out = output - output.mean(1, keepdim=True)
        delta_target = target[:, lag:, :] - target[:, lag:, :].mean(1, keepdim=True)
        var_out = delta_out.pow(2).mean(1, keepdim=True)
        var_target = delta_target.pow(2).mean(1, keepdim=True)
        corrs = (delta_out * delta_target).mean(1, keepdim=True) / (
            (var_out + self.eps) * (var_target + self.eps)
        ).sqrt()

        per_neuron_correlations = -corrs.view(-1, corrs.shape[-1]).mean(dim=0)
        self._per_neuron_correlations = per_neuron_correlations.detach()
        if self.negate:
            corrs = -corrs

        if not self.per_neuron:
            ret1 = corrs.mean() if self.avg else corrs.sum()
            ret2 = self.gamma_L1 * (pre_ca.abs().mean() if self.avg else pre_ca.abs().sum())
        else:
            ret1 = corrs.view(-1, corrs.shape[-1]).mean(dim=0)
            pre_ca = pre_ca.transpose(1, 2)
            ret2 = self.gamma_L1 * (pre_ca.view(-1, pre_ca.shape[-1]).abs().mean(dim=0))
        return ret1 + ret2


class ScaledCorrelationLoss3d(nn.Module):
    def __init__(
        self, bias: float = 1e-16, scale: float = 30.0, per_neuron: bool = False, avg: bool = False, negate: bool = True
    ):
        super().__init__()
        self.eps = bias
        self.scale = scale
        self.per_neuron = per_neuron
        self.avg = avg
        self.negate = negate
        # Placeholder to store last-computed per-neuron correlations
        self.register_buffer("_per_neuron_correlations", torch.tensor([]), persistent=False)

    def forward(self, output, target, **kwargs):
        lag = target.size(1) - output.size(1)
        corrs = torch.zeros((output.size(0), 1, output.size(2)), device=output.device)
        count = 0
        for i in range(output.size(1) // self.scale):
            delta_out = output[:, i * self.scale : (i + 1) * self.scale, :] - output[
                :, i * self.scale : (i + 1) * self.scale, :
            ].mean(1, keepdim=True)
            delta_target = target[:, lag + i * self.scale : lag + (i + 1) * self.scale, :] - target[
                :, lag + i * self.scale : lag + (i + 1) * self.scale, :
            ].mean(1, keepdim=True)
            var_out = delta_out.pow(2).mean(1, keepdim=True)
            var_target = delta_target.pow(2).mean(1, keepdim=True)
            corrs += (delta_out * delta_target).mean(1, keepdim=True) / (
                (var_out + self.eps) * (var_target + self.eps)
            ).sqrt()
            count += 1
        corrs /= count

        per_neuron_correlations = -corrs.view(-1, corrs.shape[-1]).mean(dim=0)
        self._per_neuron_correlations = per_neuron_correlations.detach()
        if self.negate:
            corrs = -corrs

        if not self.per_neuron:
            return corrs.mean() if self.avg else corrs.sum()
        else:
            return corrs.view(-1, corrs.shape[-1]).mean(dim=0)
