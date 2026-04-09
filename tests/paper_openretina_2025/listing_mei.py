from functools import partial
import torch; import matplotlib.pyplot as plt
from openretina.models import *
from openretina.insilico import *
from openretina.utils import *

model = load_core_readout_from_remote("hoefling_2024_base_low_res", "cpu")
# Define objective that increases the activity of a single neuron between timestep 10 and 20.
objective = IncreaseObjective(
    model,
    neuron_indices=[10],
    data_key="session_2_ventral2_20201016",
    response_reducer=SliceMeanReducer(axis=0, start=10, length=10),
)
# Initialize random stimulus with a variance of 0.1
stimulus = torch.randn(model.stimulus_shape(time_steps=40), requires_grad=True)
stimulus.data = stimulus.data * 0.1
# Optimize stimulus with gradient descent towards the objective.
optimize_stimulus(
    stimulus=stimulus,
    optimizer_init_fn=partial(torch.optim.SGD, lr=100.0),
    objective_object=objective,
    optimization_stopper=OptimizationStopper(max_iterations=10),
)
# Visualize stimulus
save_stimulus_to_mp4_video(stimulus[0], "mei.mp4")
fig, axes = plt.subplots(2, 2, figsize=(7 * 3, 12))
plot_stimulus_composition(stimulus[0], axes[0,0], axes[0,1], axes[1,0])

