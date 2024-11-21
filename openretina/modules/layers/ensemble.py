import torch


class EnsembleModel(torch.nn.Module):
    """An ensemble model consisting of several individual ensemble members.

    Attributes:
        *members: PyTorch modules representing the members of the ensemble.
    """

    _module_container_cls = torch.nn.ModuleList

    def __init__(self, *members: torch.nn.Module):
        """Initializes EnsembleModel."""
        super().__init__()
        self.members = self._module_container_cls(members)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Calculates the forward pass through the ensemble.

        The input is passed through all individual members of the ensemble and their outputs are averaged.

        Args:
            x: A tensor representing the input to the ensemble.
            *args: Additional arguments will be passed to all ensemble members.
            **kwargs: Additional keyword arguments will be passed to all ensemble members.

        Returns:
            A tensor representing the ensemble's output.
        """
        outputs = [m(x, *args, **kwargs) for m in self.members]
        mean_output = torch.stack(outputs, dim=0).mean(dim=0)
        return mean_output

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({', '.join(m.__repr__() for m in self.members)})"
