"""Optimize a stimulus over a subset of a model's input channels.

Some models take input channels that are not pixels. ``qiu_2026`` folds two behavior traces
(pupil size, locomotion) in as extra input channels alongside the grayscale video, so a naive
MEI optimization would happily "paint" locomotion as if it were a picture.

The obvious fix -- optimize the full tensor and reset the behavior channels after every step --
is wrong, and quietly so. ``ChangeNormJointlyClipRangeSeparately`` (see
:mod:`openretina.insilico.stimulus_optimization.regularizer`) normalizes over *all* channels
jointly. For a ``(1, 3, 50, 36, 64)`` stimulus with both behavior channels pinned at ``1.0``,
the behavior block alone has norm ``sqrt(2 * 50 * 36 * 64) = 480``,
so setting a video-sized ``norm`` of ``339`` rescales everything by ``339 / 480+ < 1``; the reset
then restores the behavior channels to full size and the next iteration shrinks the video again.
The video decays geometrically to zero. :class:`RangeRegularizationLoss` has the matching problem:
its ``max_norm`` term is computed over the whole tensor, so it stays permanently active.

This module takes the other route: hand ``optimize_stimulus`` a tensor that contains *only* the
optimized channels, and let a thin wrapper splice the constants back in on every forward pass.
Every regularizer, postprocessor and plotting helper then sees exactly the channels being
optimized, with no changes to shared code.
"""

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float


class FixedChannelStimulusModel(nn.Module):
    """Expose an ``n``-channel model as a model over a subset of its input channels.

    The remaining channels are filled with per-channel constants, broadcast over time, height and
    width on every forward pass.

    Example:
        >>> import torch
        >>> from openretina.insilico.stimulus_optimization.fixed_channel_model import (
        ...     FixedChannelStimulusModel,
        ... )
        >>> # qiu_2026: channel 0 is the video, channels 1-2 are z-scored behavior traces.
        >>> # Pin behavior to the session mean and optimize the video alone.
        >>> wrapped = FixedChannelStimulusModel(model, {1: 0.0, 2: 0.0})  # doctest: +SKIP
        >>> stimulus = torch.randn(wrapped.stimulus_shape(time_steps=50), requires_grad=True)  # doctest: +SKIP

    Args:
        model: the wrapped model. Must accept ``(batch, num_channels, t, h, w)`` and the optional
            ``data_key`` / ``pupil_center`` keyword arguments of
            :meth:`~openretina.models.core_readout.BaseCoreReadout.forward`.
        constant_channels: ``{channel_index: value}`` for every channel held fixed.
        num_channels: total input channels of the wrapped model. Inferred from
            ``model.data_info["input_shape"][0]`` when omitted.

    Note:
        The constants live in a registered buffer, not a parameter, so ``optimizer_init_fn([stimulus])``
        still sees exactly one leaf tensor and no optimizer can reach them.

        :meth:`~openretina.insilico.stimulus_optimization.objective.InnerNeuronVisualizationObjective.hook_model`
        walks ``_modules`` recursively, so through this wrapper every layer name gains a ``model_``
        prefix. That matters only for inner-neuron visualization, not for MEIs or response sweeps.
    """

    model: nn.Module
    constant_values: torch.Tensor

    def __init__(
        self,
        model: nn.Module,
        constant_channels: Mapping[int, float],
        num_channels: int | None = None,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = self._infer_num_channels(model)
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}.")

        out_of_range = sorted(c for c in constant_channels if not 0 <= c < num_channels)
        if out_of_range:
            raise ValueError(
                f"constant_channels indices {out_of_range} are outside the model's channel range [0, {num_channels})."
            )
        optimized_channels = tuple(c for c in range(num_channels) if c not in constant_channels)
        if not optimized_channels:
            raise ValueError(f"All {num_channels} channels were declared constant; there is nothing left to optimize.")

        self.model = model
        self._num_channels = num_channels
        self._constant_indices = tuple(sorted(constant_channels))
        self._optimized_channels = optimized_channels
        self.register_buffer(
            "constant_values",
            torch.tensor([float(constant_channels[c]) for c in self._constant_indices], dtype=torch.float32),
        )

    @staticmethod
    def _infer_num_channels(model: nn.Module) -> int:
        data_info = getattr(model, "data_info", None)
        input_shape = (data_info or {}).get("input_shape") if isinstance(data_info, Mapping) else None
        if input_shape is None:
            raise ValueError(
                "Could not infer num_channels: the model has no `data_info['input_shape']`. "
                "Pass num_channels explicitly."
            )
        return int(input_shape[0])

    @property
    def num_channels(self) -> int:
        """Total input channels of the wrapped model."""
        return self._num_channels

    @property
    def optimized_channels(self) -> tuple[int, ...]:
        """Indices (into the wrapped model's input) of the channels this wrapper optimizes, in order."""
        return self._optimized_channels

    @property
    def constant_channels(self) -> dict[int, float]:
        """``{channel_index: value}`` for the channels held fixed."""
        return {c: float(v) for c, v in zip(self._constant_indices, self.constant_values, strict=True)}

    def assemble(self, x: Float[torch.Tensor, "batch optimized t h w"]) -> Float[torch.Tensor, "batch channels t h w"]:
        """Splice the constant channels back in, returning the full input the wrapped model expects."""
        if x.dim() != 5:
            raise ValueError(f"Expected a 5d (batch, channels, t, h, w) stimulus, got {tuple(x.shape)}.")
        if x.shape[1] != len(self._optimized_channels):
            raise ValueError(
                f"Expected {len(self._optimized_channels)} channels in dim 1 (the optimized channels "
                f"{self._optimized_channels}), got {tuple(x.shape)}."
            )

        batch, _, time_steps, height, width = x.shape
        constant_positions = {c: i for i, c in enumerate(self._constant_indices)}
        channels: list[torch.Tensor] = []
        next_optimized = 0
        for channel in range(self._num_channels):
            if channel in constant_positions:
                value = self.constant_values[constant_positions[channel]].to(dtype=x.dtype, device=x.device)
                # `expand` rather than `repeat`: a broadcast view, so no per-iteration allocation.
                channels.append(value.view(1, 1, 1, 1, 1).expand(batch, 1, time_steps, height, width))
            else:
                channels.append(x[:, next_optimized : next_optimized + 1])
                next_optimized += 1
        return torch.cat(channels, dim=1)

    def forward(
        self,
        x: Float[torch.Tensor, "batch optimized t h w"],
        data_key: str | None = None,
        pupil_center: Float[torch.Tensor, "batch two t"] | None = None,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {}
        if data_key is not None:
            kwargs["data_key"] = data_key
        if pupil_center is not None:
            kwargs["pupil_center"] = pupil_center
        return self.model(self.assemble(x), **kwargs)

    def stimulus_shape(self, time_steps: int, num_batches: int = 1) -> tuple[int, int, int, int, int]:
        """Shape of the *optimized* stimulus: like the wrapped model's, minus the constant channels."""
        stimulus_shape = getattr(self.model, "stimulus_shape")
        batch, _, t, height, width = stimulus_shape(time_steps, num_batches)
        return batch, len(self._optimized_channels), t, height, width

    def readout_keys(self) -> list[str]:
        """Session ids of the wrapped model, so callers can key objectives without unwrapping."""
        source: Any = self.model if hasattr(self.model, "readout_keys") else getattr(self.model, "readout")
        return list(source.readout_keys())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_channels={self._num_channels}, "
            f"optimized_channels={self._optimized_channels}, constant_channels={self.constant_channels})"
        )
