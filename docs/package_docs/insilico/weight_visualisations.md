While the use of optimised stimuli can give an indication of the features that a modelled neuron responds to, it does not indicate how the model computed a feature.
Speaking in terms of retinal information processing, an MEI gives insights about the message that a retinal neuron sends to the brain, but it does not tell us about how the extraction of this message is implemented in retinal circuitry.
Towards such a mechanistic understanding, we can extend the approach of optimising stimuli to internal neurons of the model.
To the extent that the architecture of the model corresponds to the architecture of the retinal circuit (e.g., in terms of the number of layers, or the number of cell types or convolutional channels, respectively), a transfer of insights gained about internal model neurons to biological retinal interneurons is warranted (see e.g., [Maheswaranathan et al., Neuron, 2023](https://doi.org/10.1016/j.neuron.2023.06.007) and [Schröder et al., NeurIPS, 2020](https://papers.nips.cc/paper/2020/file/b139e104214a08ae3f2ebcce149cdf6e-Paper.pdf)).
For the default Core + Readout architecture, we support weight visualisation for both the convolutional layer and the readout layer, as illustrated in the following code example:

```python
from openretina.models import *
model = load_core_readout_from_remote("hoefling_2024_base_low_res", "cpu")
# Visualise a specific channel weight for the first convolutional layer
conv_layer_figure = model.core.plot_weight_visualization(
    layer=0, in_channel=1, out_channel=0)
# Visualise the readout weights of a neuron for a particular readout session
session_key = model.readout.readout_keys()[0]
readout_figure = model.readout[session_key].plot_weight_for_neuron(5)
```

For convolutional layers, we visualise the weights of each channel of a convolutional layer separately.
The spatiotemporal separable convolution layer consists of a two-dimensional spatial weight and one-dimensional temporal weight.
This temporal component is then multiplied by the spatial weight to receive a three-dimensional tensor for the
convolution operation.
The conv_layer_figure of the code example plots both the spatial weight, the computed temporal component, and the sine and cosine temporal weights.

The readout layer can be visualised as a two-dimensional Gaussian mask that defines from which position the activities are read out from the core.
Similarly, the feature weights that indicate the importance of each channel of the core for the output of the current neuron, are visualised as an one-dimensional bar chart.
The readout_figure of the code example contains both of these visualisations. 
By combining the visualisations of the weights and the MEIs of the internal neurons, we get a sense how the neural network calculates its features.

With openretina’s command line interface, you can create weight visualisations in combination with the MEIs for _all_ internal and output neurons as follows:

```bash
# Explore the options of the visualise tool
openretina visualize --help
# Download and visualise a pretrained model
openretina visualize --model-path hoefling_2024_base_low_res --save-folder visualizations
# We also support the original ensemble model of the paper Hoefling, Elife, 2024.
openretina visualize --is-hoefling-ensemble-model --model-id 0 --save-folder vis_ensemble_0
```

Here, we first print the help string of the visualize command. 
We then plot weight and MEI visualisations for all neurons of the pre-trained model “hoefling_2024_base_low_res”, which are written into the folder “visualizations”.
We also support to download and visualise the original ensemble model used in the paper by [Höfling et al., Elife, 2024](https://doi.org/10.7554/eLife.86860).
The last command will download this model and visualise all weights and MEIs of the first model of the ensemble models.