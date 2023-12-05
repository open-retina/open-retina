from torch import nn
import torch


class PoissonLoss3d(nn.Module):
    def __init__(self, bias=1e-16, per_neuron=False, avg=False):
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron
        self.avg = avg

    def forward(self, output, target):
        lag = target.size(1) - output.size(1)
        loss = output - target[:, lag:, :] * torch.log(output + self.bias)
        if not self.per_neuron:
            return loss.mean() if self.avg else loss.sum()#loss.sum(axis=(-2)).mean()
        else:
            return loss.view(-1, loss.shape[-1]).mean(dim=0)


class CelltypePoissonLoss3d(nn.Module):
    def __init__(self, bias=1e-16, per_neuron=False, avg=False):
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron
        self.avg = avg

    def forward(self, output, target, group_assignment, group_counts):
        lag = target.size(1) - output.size(1)
        loss = output - target[:, lag:, :] * torch.log(output + self.bias)
        loss = loss * (group_counts.sum() / group_counts.size(0) / group_counts[group_assignment-1])[:,None,...]
        if not self.per_neuron:
            return loss.mean() if self.avg else loss.sum()#loss.sum(axis=(-2)).mean()
        else:
            return loss.view(-1, loss.shape[-1]).mean(dim=0)


class StandardPoissonLoss3d(nn.Module): # standard poisson loss for ct dataloader
    def __init__(self, bias=1e-16, per_neuron=False, avg=False):
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron
        self.avg = avg

    def forward(self, output, target, group_assignment, group_counts):
        lag = target.size(1) - output.size(1)
        loss = output - target[:, lag:, :] * torch.log(output + self.bias)
        if not self.per_neuron:
            return loss.mean() if self.avg else loss.sum()#loss.sum(axis=(-2)).mean()
        else:
            return loss.view(-1, loss.shape[-1]).mean(dim=0)


class CorrelationLoss3d(nn.Module):
    def __init__(self, bias=1e-16, per_neuron=False, avg=False):
        super().__init__()
        self.eps = bias
        self.per_neuron = per_neuron
        self.avg = avg

    def forward(self, output, target, group_assignment, group_counts):
        lag = target.size(1) - output.size(1)
        delta_out = output - output.mean(1, keepdim=True)
        delta_target = target[:, lag:, :] - target[:, lag:, :].mean(1, keepdim=True)
        var_out = delta_out.pow(2).mean(1, keepdim=True)
        var_target = delta_target.pow(2).mean(1, keepdim=True)
        corrs = (delta_out * delta_target).mean(1, keepdim=True) / (
            (var_out + self.eps) * (var_target + self.eps)
        ).sqrt()
        if not self.per_neuron:
            return -corrs.mean() if self.avg else -corrs.sum()
        else:
            return -corrs.view(-1, corrs.shape[-1]).mean(dim=0)


class CelltypeCorrelationLoss3d(nn.Module):
    def __init__(self, bias=1e-16, per_neuron=False, avg=False):
        super().__init__()
        self.eps = bias
        self.per_neuron = per_neuron
        self.avg = avg

    def forward(self, output, target, group_assignment, group_counts):
        lag = target.size(1) - output.size(1)
        delta_out = output - output.mean(1, keepdim=True)
        delta_target = target[:, lag:, :] - target[:, lag:, :].mean(1, keepdim=True)
        var_out = delta_out.pow(2).mean(1, keepdim=True)
        var_target = delta_target.pow(2).mean(1, keepdim=True)
        corrs = (delta_out * delta_target).mean(1, keepdim=True) / (
            (var_out + self.eps) * (var_target + self.eps)
        ).sqrt()
        corrs = corrs * (group_counts.sum() / group_counts.size(0) / group_counts[group_assignment-1])[:,None,...]
        if not self.per_neuron:
            return -corrs.mean() if self.avg else -corrs.sum()
        else:
            return -corrs.view(-1, corrs.shape[-1]).mean(dim=0)


class L1CorrelationLoss3d(nn.Module):
    def __init__(self, bias=1e-16, per_neuron=False, avg=False):
        super().__init__()
        self.eps = bias
        self.per_neuron = per_neuron
        self.avg = avg
        # self.gamma_L1 = 0.001
        self.gamma_L1 = 0.0002
    def forward(self, output, target, group_assignment, group_counts):
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
        if not self.per_neuron:
            ret1 = -corrs.mean() if self.avg else -corrs.sum()
            ret2 = self.gamma_L1 * (pre_ca.abs().mean() if self.avg else pre_ca.abs().sum())
        else:
            ret1 = -corrs.view(-1, corrs.shape[-1]).mean(dim=0)
            pre_ca = pre_ca.transpose(1,2)
            ret2 =  self.gamma_L1 * (pre_ca.view(-1, pre_ca.shape[-1]).abs().mean(dim=0))
        print('corr loss', ret1)
        print('L1 loss', ret2)
        return ret1 + ret2


class ScaledCorrelationLoss3d(nn.Module):
    def __init__(self, bias=1e-16, scale=30, per_neuron=False, avg=False):
        super().__init__()
        self.eps = bias
        self.scale = scale
        self.per_neuron = per_neuron
        self.avg = avg

    def forward(self, output, target, group_assignment, group_counts):
        lag = target.size(1) - output.size(1)
        corrs = torch.zeros((output.size(0), 1, output.size(2)), device = output.device)
        count = 0
        for i in range(output.size(1)//self.scale):
            delta_out = output[:, i * self.scale: (i + 1) * self.scale, :] \
                        - output[:, i * self.scale: (i + 1) * self.scale, :].mean(1, keepdim=True)
            delta_target = target[:, lag + i * self.scale: lag + (i + 1) * self.scale, :] \
                           - target[:, lag + i * self.scale: lag + (i + 1) * self.scale, :].mean(1, keepdim=True)
            var_out = delta_out.pow(2).mean(1, keepdim=True)
            var_target = delta_target.pow(2).mean(1, keepdim=True)
            corrs += (delta_out * delta_target).mean(1, keepdim=True) \
                         / ((var_out + self.eps) * (var_target + self.eps)).sqrt()
            count += 1
        corrs /= count
        if not self.per_neuron:
            return -corrs.mean() if self.avg else -corrs.sum()
        else:
            return -corrs.view(-1, corrs.shape[-1]).mean(dim=0)


class MSE3d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        lag = target.size(1) - output.size(1)
        loss = output - target[:, lag:, :]
        return loss.pow(2).sum(dim=-1).sum()


class StandardMSE3d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, group_assignment, group_counts):
        lag = target.size(1) - output.size(1)
        loss = output - target[:, lag:, :]
        return loss.pow(2).sum(dim=-1).sum()
