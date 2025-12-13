from omegaconf import OmegaConf

def register_resolvers() -> None:

    if not OmegaConf.has_resolver("len"):
        OmegaConf.register_new_resolver("len", lambda x: len(x))
