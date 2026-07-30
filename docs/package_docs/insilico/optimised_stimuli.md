The output of the neural network contains the model’s prediction of the biological neurons’ responses to the stimulus. A natural question to ask is which stimulus activates each neuron the most by finding the maximally exciting input (MEI). As shown in the code example below, openretina supports this by defining an objective and then using the function `optimize_stimulus` to optimise a stimulus towards the given objective.

```python
from functools import partial
import torch
import matplotlib.pyplot as plt

from openretina.models import *
from openretina.insilico import *
from openretina.utils import *

# Load model on cpu
model = load_core_readout_from_remote("hoefling_2024_base_low_res", "cpu")

# Define objective that increases the activity of a single neuron 
# between timestep 10 and 20.
objective = IncreaseObjective(
model,
neuron_indices=[10],
data_key="session_2_ventral2_20201016",
response_reducer=SliceMeanReducer(axis=0, start=10, length=10),
)

# Initialise random stimulus with a variance of 0.1
stimulus = torch.randn(model.stimulus_shape(time_steps=40), requires_grad=True)
stimulus.data = stimulus.data * 0.1

# Optimise stimulus with gradient descent towards the objective.
optimize_stimulus(
    stimulus=stimulus,
    optimizer_init_fn=partial(torch.optim.SGD, lr=100.0),
    objective_object=objective,
    optimization_stopper=OptimizationStopper(max_iterations=10),
)

# Visualise stimulus
save_stimulus_to_mp4_video(stimulus[0], "mei.mp4")
fig, axes = plt.subplots(2, 2, figsize=(7 * 3, 12))
plot_stimulus_composition(stimulus[0], axes[0,0], axes[0,1], axes[1,0])
```

In this case, the pre-implemented objective maximises the response of a single neuron: we initialise a random stimulus and optimise it towards our objective using gradient descent with a learning rate of 100.0 for 10 optimisation steps.
The result is the stimulus that the model predicts to be maximally exciting for the given neuron.
This optimised stimulus can then be saved as a video, as shown in the example code, or decomposed into its temporal and spatial components; for illustration, see the accompanying Jupyter notebook [notebooks/mei_example.ipynb](https://github.com/open-retina/open-retina/blob/main/notebooks/hoefling_gradient_analysis.ipynb).
For the decomposition, we followed [Höfling et al., eLife, 2024](https://doi.org/10.7554/eLife.86860) and ran singular value decomposition on each colour channel individually.
The visualisation then shows the first singular vector for the decomposed spatial and temporal dimensions, respectively. 
MEIs can be used to gain insights about neurons’ tuning properties. 

By analysing the MEIs of a trained model, [Höfling et al.](https://doi.org/10.7554/eLife.86860) were able to discover an RGC type selective to chromatic contrasts that allows the detection of changes in visual context.
By additionally confirming these results experimentally, this provides an example of how optimised stimuli can lead to a better understanding of the features to which a neuron is selective.