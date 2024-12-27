import torch


class PoissonLoss3d(torch.nn.Module):
    def __init__(self, bias: float = 1e-16, per_neuron: bool = False, avg: bool = False):
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron
        self.avg = avg

    def forward(self, output, target):
        lag = target.size(1) - output.size(1)
        loss = output - target[:, lag:, :] * torch.log(output + self.bias)
        if not self.per_neuron:
            return loss.mean() if self.avg else loss.sum()  # loss.sum(axis=(-2)).mean()
        else:
            return loss.view(-1, loss.shape[-1]).mean(dim=0)

    def __str__(self):
        bias, per_neuron, avg = self.bias, self.per_neuron, self.avg
        return f"PoissonLoss3d({bias=} {per_neuron=} {avg=})"


class CelltypePoissonLoss3d(torch.nn.Module):
    def __init__(self, bias: float = 1e-16, per_neuron: bool = False, avg: bool = False):
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron
        self.avg = avg

    def forward(self, output, target, group_assignment, group_counts):
        lag = target.size(1) - output.size(1)
        loss = output - target[:, lag:, :] * torch.log(output + self.bias)
        loss = loss * (group_counts.sum() / group_counts.size(0) / group_counts[group_assignment - 1])[:, None, ...]
        if not self.per_neuron:
            return loss.mean() if self.avg else loss.sum()  # loss.sum(axis=(-2)).mean()
        else:
            return loss.view(-1, loss.shape[-1]).mean(dim=0)
