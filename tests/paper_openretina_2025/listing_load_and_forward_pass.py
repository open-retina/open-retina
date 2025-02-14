import torch
from openretina.models import *
# Download model
model = load_core_readout_from_remote("hoefling_2024_base_low_res", "cpu")
# Run forward pass
responses = model.forward(torch.rand(model.stimulus_shape(time_steps=50)))
