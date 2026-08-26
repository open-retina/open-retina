"""Sweep a model's behavioral inputs and record how its response changes.

``qiu_2026`` gives a model two distinct behavioral pathways, and this module probes both:

1. **Behavior channels.** Pupil size and locomotion enter as extra *input channels*, broadcast
   constant over height and width (``data_io/qiu_2026/stimuli.py``). They therefore reach the core
   only through the first convolution's input weights -- a constant per-output-channel offset ahead
   of BatchNorm and ELU. A state-dependent bias before a nonlinearity is the only mechanism this
   architecture has for behavioral gain: it can rescale or threshold a response, but it cannot
   reshape a receptive field. :func:`behavior_response_grid` measures that gain.
2. **The pupil shifter.** Pupil *position* goes through a per-session MLP whose output is added to
   every neuron's readout grid location, so it genuinely *moves* the receptive field.
   :func:`pupil_center_response_grid` measures the response cost of that movement, and
   :func:`shifter_shift_grid` reads the displacement straight off the MLP.

All sweeps are z-scored per session upstream, so ``0.0`` is that session's mean state and the
default grid spans +/-2 standard deviations.

.. warning::
    A 2-D grid necessarily visits combinations the animal never produced (pupil ``+2`` with
    locomotion ``-2``). Overlay the empirical joint density -- for ``qiu_2026`` that is
    ``movies[session].train[1:3, :, 0, 0]`` -- before reading anything into the corners.
"""

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Float

from openretina.insilico.stimulus_optimization.objective import ResponseReducer

#: Default sweep grid: +/-2 standard deviations of the z-scored behavior traces, in steps of 0.25.
DEFAULT_SWEEP_VALUES = np.arange(-2.0, 2.25, 0.25)


@dataclass
class ResponseGrid:
    """Responses (and optionally their gradients) over a 2-D parameter grid.

    Attributes:
        axis_values: the swept values along each axis, ``(n_0,)`` and ``(n_1,)``.
        axis_names: human-readable axis labels, for plotting.
        responses: ``(n_0, n_1, neurons)`` reduced response at every grid point.
        gradients: ``(2, n_0, n_1, neurons)`` finite-difference gradient, or None.
    """

    axis_values: tuple[np.ndarray, np.ndarray]
    axis_names: tuple[str, str]
    responses: Float[np.ndarray, "n_0 n_1 neurons"]
    gradients: Float[np.ndarray, "two n_0 n_1 neurons"] | None = None

    @property
    def n_neurons(self) -> int:
        return int(self.responses.shape[-1])

    def for_neuron(self, index: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Return ``(responses, gradients)`` for one neuron, laid out for the vector-field plots.

        The shapes -- ``(n_0, n_1)`` and ``(2, n_0, n_1)`` -- are exactly what
        :func:`openretina.utils.plotting.plot_vector_field_resp_iso` expects for its
        ``resp_dict`` and ``gradient_dict`` arguments.
        """
        gradients = None if self.gradients is None else self.gradients[:, :, :, index]
        return self.responses[:, :, index], gradients

    def save(self, path: str | os.PathLike) -> Path:
        """Write this grid to a ``.npz`` so it can be re-plotted without recomputing it.

        Deliberately plain arrays rather than a pickle: the file stays loadable if this class ever
        changes, and :meth:`load` can refuse pickles outright. The ``.npz`` suffix is added if
        missing, and parent directories are created.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "axis_values_0": np.asarray(self.axis_values[0]),
            "axis_values_1": np.asarray(self.axis_values[1]),
            "axis_names": np.asarray(self.axis_names),
            "responses": np.asarray(self.responses),
        }
        if self.gradients is not None:
            arrays["gradients"] = np.asarray(self.gradients)
        # allow_pickle=False refuses to *write* anything but plain arrays, so a file produced
        # here is always safe for `load` to read back with pickling disabled.
        np.savez_compressed(path, allow_pickle=False, **arrays)
        return path

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ResponseGrid":
        """Read back a grid written by :meth:`save`.

        Loads with ``allow_pickle=False``, so a tampered-with file cannot execute code.
        """
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                axis_values=(data["axis_values_0"], data["axis_values_1"]),
                axis_names=(str(data["axis_names"][0]), str(data["axis_names"][1])),
                responses=data["responses"],
                gradients=data["gradients"] if "gradients" in data.files else None,
            )

    def modulation_index(self) -> Float[np.ndarray, " neurons"]:
        """``(max - min) / (max + min)`` over the grid, per neuron.

        Zero means the sweep left the neuron untouched; larger values mean stronger behavioral
        modulation. Neurons whose responses sum to zero over the grid (or straddle zero so that
        ``max + min == 0``) yield ``nan`` rather than an infinity -- the index is only meaningful
        for a non-negative response, which is what the models' output nonlinearities produce.
        """
        flat = self.responses.reshape(-1, self.responses.shape[-1])
        highest, lowest = flat.max(axis=0), flat.min(axis=0)
        denominator = highest + lowest
        return np.divide(
            highest - lowest,
            denominator,
            out=np.full_like(denominator, np.nan, dtype=np.float64),
            where=denominator != 0,
        )


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _as_grid_values(values: Sequence[float] | np.ndarray | None) -> np.ndarray:
    grid = np.asarray(DEFAULT_SWEEP_VALUES if values is None else values, dtype=np.float64).ravel()
    if grid.size < 2:
        raise ValueError(f"A sweep axis needs at least 2 values, got {grid.size}.")
    return grid


def _finite_difference_gradients(
    responses: Float[np.ndarray, "n_0 n_1 neurons"], axis_values: tuple[np.ndarray, np.ndarray]
) -> Float[np.ndarray, "two n_0 n_1 neurons"]:
    """Gradients along both sweep axes, by central differences on the grid we already computed.

    Per-point autograd would cost ``n_0 * n_1`` backward passes through the whole model for every
    neuron; on the real qiu model that is 289 x 17.3k. Finite differences are free here because the
    grid exists anyway, and they are exact for the linear case. If you ever need true per-point
    derivatives, ``torch.func.jvp`` over the two behavior values is the cheap way to get them.
    """
    derivatives = np.gradient(responses, *axis_values, axis=(0, 1))
    return np.stack(derivatives, axis=0)


@torch.no_grad()
def _evaluate_grid(
    model: torch.nn.Module,
    data_key: str | None,
    axis_values: tuple[np.ndarray, np.ndarray],
    make_point: Callable[[float, float], tuple[torch.Tensor, torch.Tensor | None]],
    response_reducer: ResponseReducer,
    neuron_indices: Sequence[int] | None,
    grid_batch_size: int,
    device: torch.device | str | None,
) -> Float[np.ndarray, "n_0 n_1 neurons"]:
    """Evaluate ``make_point`` over the full grid, in chunks, and reduce each response over time."""
    if model.training:
        raise ValueError(
            "The model is in training mode. A sampled Gaussian readout draws its grid locations "
            "stochastically while training, which adds a noise floor to every point of the sweep "
            "and makes the result irreproducible. Call model.eval() first."
        )
    if grid_batch_size < 1:
        raise ValueError(f"grid_batch_size must be >= 1, got {grid_batch_size}.")

    values_0, values_1 = axis_values
    if device is None:
        device = _model_device(model)
    points = [(i, j) for i in range(len(values_0)) for j in range(len(values_1))]

    responses: np.ndarray | None = None
    for start in range(0, len(points), grid_batch_size):
        chunk = points[start : start + grid_batch_size]
        built = [make_point(float(values_0[i]), float(values_1[j])) for i, j in chunk]
        stimulus_batch = torch.stack([stimulus for stimulus, _ in built]).to(device)

        kwargs: dict[str, Any] = {}
        if data_key is not None:
            kwargs["data_key"] = data_key
        pupils = [pupil for _, pupil in built if pupil is not None]
        if pupils:
            kwargs["pupil_center"] = torch.stack(pupils).to(device)

        output = model(stimulus_batch, **kwargs)  # (batch, t_out, neurons)
        if neuron_indices is not None:
            output = output[..., list(neuron_indices)]
        # `response_reducer` reduces over axis 0, matching every other caller, so feed it one
        # (t_out, neurons) response at a time rather than the batched tensor.
        reduced = torch.stack([response_reducer.forward(output[b]) for b in range(output.shape[0])])

        if responses is None:
            responses = np.empty((len(values_0), len(values_1), reduced.shape[-1]), dtype=np.float64)
        for (i, j), row in zip(chunk, reduced.detach().cpu().numpy(), strict=True):
            responses[i, j] = row

    assert responses is not None  # the grid always has at least 2x2 points
    return responses


def _expand_pupil_center(
    pupil_center: Sequence[float] | np.ndarray | torch.Tensor | None, time_steps: int
) -> torch.Tensor | None:
    """``(2,)`` -> ``(2, t)``, held constant over the whole clip. ``None`` passes through."""
    if pupil_center is None:
        return None
    center = torch.as_tensor(np.asarray(pupil_center, dtype=np.float32)).reshape(2)
    return center[:, None].expand(2, time_steps).contiguous()


def _as_video(video: torch.Tensor) -> torch.Tensor:
    video = torch.as_tensor(video, dtype=torch.float32)
    if video.dim() == 3:
        video = video.unsqueeze(0)
    if video.dim() != 4:
        raise ValueError(f"Expected a (t, h, w) or (channels, t, h, w) video, got shape {tuple(video.shape)}.")
    return video


def behavior_response_grid(
    model: torch.nn.Module,
    data_key: str | None,
    video: Float[torch.Tensor, "t h w"],
    response_reducer: ResponseReducer,
    behavior_values: Sequence[float] | np.ndarray | None = None,
    behavior_channels: tuple[int, int] = (1, 2),
    axis_names: tuple[str, str] = ("pupil size [z]", "locomotion [z]"),
    pupil_center: Sequence[float] | None = None,
    neuron_indices: Sequence[int] | None = None,
    compute_gradients: bool = True,
    grid_batch_size: int = 8,
    device: torch.device | str | None = None,
) -> ResponseGrid:
    """Response to one fixed video across a 2-D grid of behavior-channel values.

    Args:
        model: a model in eval mode taking ``(batch, channels, t, h, w)``.
        data_key: session id.
        video: the video to hold fixed, ``(t, h, w)`` or ``(channels, t, h, w)``. The behavior
            channels are inserted around it, so this carries only the actual video channel(s).
        response_reducer: reduces a ``(t_out, neurons)`` response to ``(neurons,)``.
        behavior_values: values swept along *both* axes. Defaults to
            :data:`DEFAULT_SWEEP_VALUES` (-2 to +2 in steps of 0.25).
        behavior_channels: input-channel indices of the two swept behavior traces, in axis order.
        axis_names: labels for those two axes.
        pupil_center: eye position held fixed for the whole sweep, in the shifter's units.
            ``(0.0, 0.0)`` is the session mean; ``None`` omits the shifter entirely. Keep this
            consistent with whatever was used to optimize the stimulus being probed.
        neuron_indices: restrict the output to these neurons (applied after the forward pass).
        compute_gradients: also return finite-difference gradients over the grid.
        grid_batch_size: grid points evaluated per forward pass.
        device: defaults to the model's own device.
    """
    video = _as_video(video)
    n_video_channels, time_steps, height, width = video.shape
    num_channels = n_video_channels + len(behavior_channels)
    if len(set(behavior_channels)) != len(behavior_channels):
        raise ValueError(f"behavior_channels must be distinct, got {behavior_channels}.")
    out_of_range = [c for c in behavior_channels if not 0 <= c < num_channels]
    if out_of_range:
        raise ValueError(
            f"behavior_channels {out_of_range} are outside the model's channel range [0, {num_channels}); "
            f"the video contributes {n_video_channels} channel(s)."
        )
    video_channels = [c for c in range(num_channels) if c not in behavior_channels]

    grid_values = _as_grid_values(behavior_values)
    pupil = _expand_pupil_center(pupil_center, time_steps)

    def make_point(value_0: float, value_1: float) -> tuple[torch.Tensor, torch.Tensor | None]:
        stimulus = torch.empty((num_channels, time_steps, height, width), dtype=torch.float32)
        stimulus[video_channels] = video
        stimulus[behavior_channels[0]] = value_0
        stimulus[behavior_channels[1]] = value_1
        return stimulus, pupil

    axis_values = (grid_values, grid_values.copy())
    responses = _evaluate_grid(
        model, data_key, axis_values, make_point, response_reducer, neuron_indices, grid_batch_size, device
    )
    gradients = _finite_difference_gradients(responses, axis_values) if compute_gradients else None
    return ResponseGrid(axis_values=axis_values, axis_names=axis_names, responses=responses, gradients=gradients)


def pupil_center_response_grid(
    model: torch.nn.Module,
    data_key: str,
    stimulus: Float[torch.Tensor, "channels t h w"],
    response_reducer: ResponseReducer,
    pupil_values: Sequence[float] | np.ndarray | None = None,
    axis_names: tuple[str, str] = ("pupil x [z]", "pupil y [z]"),
    neuron_indices: Sequence[int] | None = None,
    compute_gradients: bool = True,
    grid_batch_size: int = 8,
    device: torch.device | str | None = None,
) -> ResponseGrid:
    """Response to one fixed stimulus across a 2-D grid of eye positions.

    Unlike :func:`behavior_response_grid` the stimulus never changes here; only the shifter's input
    does, which moves every neuron's readout location. Axis 0 is the shifter's first input feature
    and axis 1 its second, matching the ``pupil_center`` layout used during training.

    Args:
        stimulus: the *complete* model input, ``(channels, t, h, w)`` -- behavior channels included,
            since they are held fixed here.

    Note:
        The shift is added *after* ``mu`` is clamped to ``[-1, 1]``
        (:mod:`openretina.modules.readout.gaussian`), so a large enough shift pushes a neuron off
        the feature map and its response collapses. That plateau is a property of the readout, not
        a tuning curve.
    """
    if getattr(model, "shifter", None) is None:
        raise ValueError(
            "The model has no shifter, so eye position cannot affect its response. "
            "Use behavior_response_grid for the behavior input channels instead."
        )
    stimulus = torch.as_tensor(stimulus, dtype=torch.float32)
    if stimulus.dim() != 4:
        raise ValueError(f"Expected a (channels, t, h, w) stimulus, got shape {tuple(stimulus.shape)}.")
    time_steps = stimulus.shape[1]
    grid_values = _as_grid_values(pupil_values)

    def make_point(value_0: float, value_1: float) -> tuple[torch.Tensor, torch.Tensor | None]:
        pupil = _expand_pupil_center((value_0, value_1), time_steps)
        return stimulus, pupil

    axis_values = (grid_values, grid_values.copy())
    responses = _evaluate_grid(
        model, data_key, axis_values, make_point, response_reducer, neuron_indices, grid_batch_size, device
    )
    gradients = _finite_difference_gradients(responses, axis_values) if compute_gradients else None
    return ResponseGrid(axis_values=axis_values, axis_names=axis_names, responses=responses, gradients=gradients)


@torch.no_grad()
def shifter_shift_grid(
    model: torch.nn.Module,
    data_keys: Sequence[str] | None = None,
    pupil_values: Sequence[float] | np.ndarray | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Float[np.ndarray, "n_0 n_1 two"]]:
    """Evaluate the pupil shifter itself over a grid of eye positions, per session.

    This is a pure MLP evaluation -- no core, no readout, milliseconds for every session at once --
    and it answers "how far does the shifter actually move a receptive field?" directly, without
    the confound of the response.

    Returns:
        ``{session: (n_0, n_1, 2)}`` shifts in normalized readout-grid units. Last-axis index 0 is
        the **width** displacement and index 1 the **height** displacement, because
        ``F.grid_sample`` reads ``grid[..., 0]`` as the x coordinate. (The readout's own local
        variables name those dimensions ``w_in, h_in`` in the opposite order; the sampling
        convention, not the naming, is what holds.) Convert to pixels with
        :func:`shift_to_core_pixels`.
    """
    shifter = getattr(model, "shifter", None)
    if shifter is None:
        raise ValueError("The model has no shifter.")
    if data_keys is None:
        data_keys = list(shifter.keys())
    if device is None:
        device = _model_device(model)

    grid_values = _as_grid_values(pupil_values)
    mesh_0, mesh_1 = np.meshgrid(grid_values, grid_values, indexing="ij")
    positions = torch.as_tensor(np.stack([mesh_0.ravel(), mesh_1.ravel()], axis=-1), dtype=torch.float32, device=device)

    shifts: dict[str, np.ndarray] = {}
    for key in data_keys:
        flat = shifter(positions, key)
        shifts[key] = flat.detach().cpu().numpy().reshape(len(grid_values), len(grid_values), 2)
    return shifts


def shift_to_core_pixels(
    shift: Float[np.ndarray, "... two"],
    core_output_shape: Sequence[int],
    align_corners: bool = True,
) -> Float[np.ndarray, "... two"]:
    """Convert a normalized readout-grid shift to a displacement in core-output pixels.

    Args:
        shift: ``(..., 2)`` in normalized ``[-1, 1]`` grid units, last axis ordered
            ``(width, height)`` as returned by :func:`shifter_shift_grid`.
        core_output_shape: the core's output spatial shape, ordered ``(height, width)``; the last
            two entries of a longer shape are used, so a full ``(c, t, h, w)`` also works.
        align_corners: must match the readout's own ``align_corners``. With ``True`` the extremes
            ``-1`` and ``+1`` land on pixel centres ``0`` and ``n - 1``; with ``False`` they land on
            the outer edges ``-0.5`` and ``n - 0.5``.

    Returns:
        The same shape, in pixels, last axis ordered ``(width, height)``.
    """
    shift = np.asarray(shift, dtype=np.float64)
    if shift.shape[-1] != 2:
        raise ValueError(f"Expected the last axis of `shift` to have length 2, got {shift.shape[-1]}.")
    if len(core_output_shape) < 2:
        raise ValueError(f"core_output_shape needs at least 2 entries (height, width), got {tuple(core_output_shape)}.")
    height, width = int(core_output_shape[-2]), int(core_output_shape[-1])
    span = np.array([width - 1, height - 1] if align_corners else [width, height], dtype=np.float64)
    return shift * span / 2.0
