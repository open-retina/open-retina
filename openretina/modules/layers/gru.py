import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    """
    Convolutional GRU cell from: https://github.com/sinzlab/Sinz2018_NIPS/blob/master/nips2018/architectures/cores.py
    """

    def __init__(
        self,
        input_channels,
        rec_channels,
        input_kern: int,
        rec_kern: int,
        groups: int = 1,
        gamma_rec: int = 0,
        pad_input: bool = True,
        **kwargs,
    ):
        super().__init__()

        input_padding = input_kern // 2 if pad_input else 0
        rec_padding = rec_kern // 2

        self.rec_channels = rec_channels
        self._shrinkage = 0 if pad_input else input_kern - 1
        self.groups = groups

        self.gamma_rec = gamma_rec
        self.reset_gate_input = nn.Conv2d(
            input_channels,
            rec_channels,
            input_kern,
            padding=input_padding,
            groups=self.groups,
        )
        self.reset_gate_hidden = nn.Conv2d(
            rec_channels,
            rec_channels,
            rec_kern,
            padding=rec_padding,
            groups=self.groups,
        )

        self.update_gate_input = nn.Conv2d(
            input_channels,
            rec_channels,
            input_kern,
            padding=input_padding,
            groups=self.groups,
        )
        self.update_gate_hidden = nn.Conv2d(
            rec_channels,
            rec_channels,
            rec_kern,
            padding=rec_padding,
            groups=self.groups,
        )

        self.out_gate_input = nn.Conv2d(
            input_channels,
            rec_channels,
            input_kern,
            padding=input_padding,
            groups=self.groups,
        )
        self.out_gate_hidden = nn.Conv2d(
            rec_channels,
            rec_channels,
            rec_kern,
            padding=rec_padding,
            groups=self.groups,
        )

        self.apply(self.init_conv)
        self.register_parameter("_prev_state", None)

    def init_state(self, input_):
        batch_size, _, *spatial_size = input_.data.size()
        state_size = [batch_size, self.rec_channels] + [s - self._shrinkage for s in spatial_size]
        prev_state = torch.zeros(*state_size)
        if input_.is_cuda:
            prev_state = prev_state.cuda()
        prev_state = nn.Parameter(prev_state)
        return prev_state

    def forward(self, input_, prev_state):
        # get batch and spatial sizes

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = self.init_state(input_)

        update_gate = self.update_gate_input(input_) + self.update_gate_hidden(prev_state)
        update_gate = F.sigmoid(update_gate)

        reset_gate = self.reset_gate_input(input_) + self.reset_gate_hidden(prev_state)
        reset_gate = F.sigmoid(reset_gate)

        out = self.out_gate_input(input_) + self.out_gate_hidden(prev_state * reset_gate)
        h_t = F.tanh(out)
        new_state = prev_state * (1 - update_gate) + h_t * update_gate

        return new_state

    def regularizer(self):
        return self.gamma_rec * self.bias_l1()

    def bias_l1(self):
        return (
            self.reset_gate_hidden.bias.abs().mean() / 3  # type: ignore
            + self.update_gate_hidden.weight.abs().mean() / 3
            + self.out_gate_hidden.bias.abs().mean() / 3  # type: ignore
        )

    def __repr__(self) -> str:
        s = super().__repr__()
        s += f" [{self.__class__.__name__} regularizers: "
        ret = [
            f"{attr} = {getattr(self, attr)}"
            for attr in filter(lambda x: not x.startswith("_") and "gamma" in x, dir(self))
        ]
        return s + "|".join(ret) + "]\n"

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight.data)
            if m.bias is not None:
                nn.init.constant(m.bias.data, 0.0)


class GRU_Module(nn.Module):
    def __init__(
        self,
        input_channels,
        rec_channels,
        input_kern,
        rec_kern,
        groups: int = 1,
        gamma_rec: int = 0,
        pad_input: bool = True,
        **kwargs,
    ):
        """
        A GRU module for video data to add between the core and the readout.
        Receives as input the output of a 3Dcore. Expected dimensions:
            - (Batch, Channels, Frames, Height, Width) or (Channels, Frames, Height, Width)
        The input is fed sequentially to a convolutional GRU cell, based on the frames channel.
        The output has the same dimensions as the input.
        """
        super().__init__()
        self.gru = ConvGRUCell(
            input_channels,
            rec_channels,
            input_kern,
            rec_kern,
            groups=groups,
            gamma_rec=gamma_rec,
            pad_input=pad_input,
        )

    def forward(self, input_):
        """
        Forward pass definition based on
        https://github.com/sinzlab/Sinz2018_NIPS/blob/3a99f7a6985ae8dec17a5f2c54f550c2cbf74263/nips2018/architectures/cores.py#L556
        Modified to also accept 4 dimensional inputs (assuming no batch dimension is provided).
        """
        x, data_key = input_
        if len(x.shape) not in [4, 5]:
            raise RuntimeError(
                f"Expected 4D (unbatched) or 5D (batched) input to ConvGRUCell, but got input of size: {x.shape}"
            )

        batch = True
        if len(x.shape) == 4:
            batch = False
            x = torch.unsqueeze(x, dim=0)

        states = []
        hidden = None
        frame_pos = 2

        for frame in range(x.shape[frame_pos]):
            slice_channel = [frame if frame_pos == i else slice(None) for i in range(len(x.shape))]
            hidden = self.gru(x[slice_channel], hidden)
            states.append(hidden)
        out = torch.stack(states, frame_pos)
        if not batch:
            out = torch.squeeze(out, dim=0)
        return out
