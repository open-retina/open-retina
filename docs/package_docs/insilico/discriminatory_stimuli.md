To give an example of a more complex optimisation objective in our repository, we provide the Jupyter notebook
notebooks/most_discriminative_stimulus.ipynb.
In this notebook, we show how to generate stimuli that elicit maximally distinct responses in different cell types.
[Burg et al., ICLR, 2024](https://openreview.net/forum?id=9W6KaAcYlr) used this approach to cluster neurons
based on their functional responses.
Since you can define arbitrarily complex objectives as long as they are expressible as differentiable functions, this provides a powerful interface for probing the neural network.