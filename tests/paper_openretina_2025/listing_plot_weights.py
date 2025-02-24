from openretina.models import *
model = load_core_readout_from_remote("hoefling_2024_base_low_res", "cpu")
# Visualize a specific channel weight for the first convolutional layer
conv_layer_figure = model.core.plot_weight_visualization(layer=0, in_channel=1, out_channel=0)
# Visualize the readout weights of a neuron for a particular readout session
session_key = model.readout.readout_keys()[0]
readout_figure = model.readout[session_key].plot_weight_for_neuron(5)

