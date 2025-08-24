import torch
from torch import nn


class ParametrizedELU(nn.Module):
    def __init__(self, a=1.0, b=1.0, c=1.0, device="cuda", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = torch.nn.Parameter(torch.tensor(a, device=device), requires_grad=True)
        self.b = torch.nn.Parameter(torch.tensor(b, device=device), requires_grad=True)
        self.c = torch.nn.Parameter(torch.tensor(c, device=device), requires_grad=True)

    def forward(self, x):
        return torch.where(x > 0, self.c * x, self.a * (torch.exp(x / self.b) - 1))


class ParametrizedSoftplus(nn.Module):
    def __init__(self, a=1.0, b=0.0, w=1.0, learn_a=False):
        super().__init__()
        if learn_a:
            self.a = nn.Parameter(torch.tensor(a), requires_grad=True)
        else:
            self.a = torch.tensor(a)
        self.b = nn.Parameter(torch.tensor(b), requires_grad=True)
        self.w = nn.Parameter(torch.tensor(w), requires_grad=True)

    def forward(self, x):
        return self.w * torch.log(1 + torch.exp(self.a * x + self.b))
