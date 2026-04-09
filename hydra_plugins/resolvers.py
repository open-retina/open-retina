"""
This file is used to register custom resolvers (functions that can be used in the config)
for the Hydra config system. We take advantage of the fact that the `hydra_plugins` path are automatically
loaded by Hydra before configuration parsing. For more on this, see this discussion on
the `Hydra forums <https://github.com/facebookresearch/hydra/issues/2835>`.
Credits to @alexrgilbert.
"""

from omegaconf import OmegaConf

OmegaConf.register_new_resolver(name="len", resolver=lambda x: len(x), replace=True)
