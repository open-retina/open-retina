# What do we want to provide here:

# An interface that abstracts the model and returns a single (?) score per neuron
# An implementation of that interface for the 3D model? (alternatively let the models decide for themselves)
# we'll also need the initial shape of the for the stimulus to optimize, maybe also initialization

# An optimizer that gets the model and a list of neurons to optimize (?), and a loss (?)
# - it initializes a stimulus
# - then optimizes the stimulus towards the loss
# - returns the optimized stimulus
# Additionally:
# - Regularization?, postprocessing to clip the stimulus range during optimization
# - Stoppers, learning rate scheduling (probably using

# Questions:
# - How to combine the loss and the model to figure out which neurons to compute and use for the loss?
# - Gradient clipping?


# Previous implementations:
# - Sinzlab https://github.com/sinzlab/mei (good implementation, but many indirections via dynamic imports which makes the code difficult to understand
# - - MEI class: https://github.com/sinzlab/mei/blob/4bf43dd0e47d6808157466e76add099d4db353a5/mei/optimization.py#L39
# - - objective: https://github.com/sinzlab/mei/blob/4bf43dd0e47d6808157466e76add099d4db353a5/mei/objectives.py#L15
# - Controversial stimuli:
# - - Optimization: https://github.com/MaxFBurg/controversial-stimuli/blob/960c580ca03101cdd47c4e7b64cdb2c7aaf07108/controversialstimuli/optimization/torch_transform_image.py#L35
# - - Objective: https://github.com/MaxFBurg/controversial-stimuli/blob/960c580ca03101cdd47c4e7b64cdb2c7aaf07108/controversialstimuli/optimization/controversial_objectives.py#L21