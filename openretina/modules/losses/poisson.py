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


class L1PoissonLoss3d(torch.nn.Module):
    """
    Poisson Loss for 3D data with L1 regularization on the output.
    Useful for models predicting sparse firing rates.
    """

    def __init__(self, bias: float = 1e-16, per_neuron: bool = False, avg: bool = False, gamma_l1: float = 0.001):
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron
        self.avg = avg
        self.gamma_l1 = gamma_l1

    def forward(self, output, target):
        lag = target.size(1) - output.size(1)
        loss = output - target[:, lag:, :] * torch.log(output + self.bias)

        # Compute L1 regularization on output to encourage sparsity
        if not self.per_neuron:
            poisson_loss = loss.mean() if self.avg else loss.sum()
            l1_loss = self.gamma_l1 * (output.abs().mean() if self.avg else output.abs().sum())
            return poisson_loss + l1_loss
        else:
            # L1 norm per neuron
            poisson_loss = loss.view(-1, loss.shape[-1]).mean(dim=0)
            l1_loss = self.gamma_l1 * (output.view(-1, output.shape[-1]).abs().mean(dim=0))
            return poisson_loss + l1_loss

    def __str__(self):
        bias, per_neuron, avg, gamma_l1 = self.bias, self.per_neuron, self.avg, self.gamma_l1
        return f"L1PoissonLoss3d({bias=} {per_neuron=} {avg=} {gamma_l1=})"


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
