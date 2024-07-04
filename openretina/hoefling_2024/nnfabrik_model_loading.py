"""
Functions from nn_fabrik to import models and load state dicts
"""

import os
import pickle
from copy import deepcopy
from functools import partial
from importlib import import_module
from typing import Tuple, Optional

import torch
import torch.nn as nn
import yaml

from openretina.utils.misc import SafeLoaderWithTuple, tuple_constructor


def split_module_name(abs_class_name: str) -> Tuple[str, str]:
    abs_module_path = ".".join(abs_class_name.split(".")[:-1])
    class_name = abs_class_name.split(".")[-1]
    return abs_module_path, class_name


def dynamic_import(abs_module_path, class_name):
    module_object = import_module(abs_module_path)
    target_class = getattr(module_object, class_name)
    return target_class


def resolve_fn(fn_name, default_base):
    """
    Given a string `fn_name`, resolves the name into a callable object. If the name has multiple
    `.` separated parts, treat all but the last as module names to trace down to the final name.
    If just the name is given, tries to resolve the name in the `default_base` module name context
    with direct eval of `{default_base}.{fn_name}` in this function's context.

    Raises `NameError` if no object matching the name is found and `TypeError` if the resolved object is not callable.

    When successful, returns the resolved, callable object.
    """
    module_path, class_name = split_module_name(fn_name)

    try:
        fn_obj = (
            dynamic_import(module_path, class_name) if module_path else eval("{}.{}".format(default_base, class_name))
        )
    except NameError:
        raise NameError("Function `{}` does not exist".format(class_name))

    if not callable(fn_obj):
        raise TypeError("The object named {} is not callable.".format(class_name))

    return fn_obj


resolve_model = partial(resolve_fn, default_base="models")


def get_model(
    model_fn,
    model_config,
    dataloaders=None,
    seed=None,
    state_dict=None,
    strict=True,
    data_info=None,
):
    """
    Resolves `model_fn` and invokes the resolved function with `model_config` keyword arguments
    as well as the `dataloader` and `seed`. Note that the resolved `model_fn` is expected to
    accept the `dataloader` as the first positional argument and `seed` as a keyword argument.
    If you pass in `state_dict`, the resulting nn.Module instance will be loaded with the
    state_dict, using appropriate `strict` mode for loading.

    Args:
        model_fn: string name of the model builder function path to be resolved.
        Alternatively, you can pass in a callable object and no name resolution will be performed.
        model_config: a dictionary containing keyword arguments to be passed into the resolved `model_fn`
        dataloaders: a dictionary of dataloaders to be passed into the resolved `model_fn`
                     as the first positional argument
        seed: randomization seed to be passed in to as a keyword argument into the resolved `model_fn`
        state_dict: If provided, the resulting nn.Module object will be loaded with the state_dict before being returned
        strict: Controls the `strict` mode of nn.Module.load_state_dict

    Returns:
        Resulting nn.Module object.
    """

    if isinstance(model_fn, str):
        model_fn = resolve_model(model_fn)

    net = (
        model_fn(dataloaders, seed=seed, **model_config)
        if data_info is None
        else model_fn(dataloaders, data_info=data_info, seed=seed, **model_config)
    )

    if state_dict is not None:
        ignore_missing = model_config.get("transfer", False) or not strict
        load_state_dict(
            net,
            state_dict,
            match_names=model_config.get("transfer", False),
            ignore_unused=model_config.get("transfer", False),
            ignore_dim_mismatch=model_config.get("transfer", False),
            ignore_missing=ignore_missing,
        )  # we want the most flexible loading in the case of transfer

    return net


def load_state_dict(
    model,
    state_dict: dict,
    ignore_missing: bool = False,
    ignore_unused: bool = False,
    match_names: bool = False,
    ignore_dim_mismatch: bool = False,
    prefix_agreement: float = 0.98,
):
    """
    Loads given state_dict into model, but allows for some more flexible loading.

    :param model: nn.Module object
    :param state_dict: dictionary containing a whole state of the module (result of `some_model.state_dict()`)
    :param ignore_missing: if True ignores entries present in model but not in `state_dict`
    :param match_names: if True tries to match names in `state_dict` and `model.state_dict()`
                        by finding and removing a common prefix from the keys in each dict
    :param ignore_dim_mismatch: if True ignores parameters in `state_dict` that don't fit the shape in `model`
    """
    model_dict = model.state_dict()
    # 0. Try to match names by adding or removing prefix:
    if match_names:
        # find prefix keys of each state dict:
        s_pref, s_idx = find_prefix(list(state_dict.keys()), p_agree=prefix_agreement)
        m_pref, m_idx = find_prefix(list(model_dict.keys()), p_agree=prefix_agreement)
        # switch prefixes:
        stripped_state_dict = {}
        for k, v in state_dict.items():
            if k.split(".")[:s_idx] == s_pref.split("."):
                stripped_key = ".".join(k.split(".")[s_idx:])
            else:
                stripped_key = k
            new_key = m_pref + "." + stripped_key if m_pref else stripped_key
            stripped_state_dict[new_key] = v
        state_dict = stripped_state_dict

    # 1. filter out missing keys
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    unused = set(state_dict.keys()) - set(filtered_state_dict.keys())
    if unused and ignore_unused:
        print("Ignored unnecessary keys in pretrained dict:\n" + "\n".join(unused))
    elif unused:
        raise RuntimeError("Error in loading state_dict: Unused keys:\n" + "\n".join(unused))
    missing = set(model_dict.keys()) - set(filtered_state_dict.keys())
    if missing and ignore_missing:
        print("Ignored Missing keys:\n" + "\n".join(missing))
    elif missing:
        raise RuntimeError("Error in loading state_dict: Missing keys:\n" + "\n".join(missing))

    # 2. overwrite entries in the existing state dict
    updated_model_dict = {}
    for k, v in filtered_state_dict.items():
        if v.shape != model_dict[k].shape:
            if ignore_dim_mismatch:
                print("Ignored shape-mismatched parameter:", k)
                continue
            else:
                raise RuntimeError("Error in loading state_dict: Shape-mismatch for key {}".format(k))
        updated_model_dict[k] = v

    # 3. load the new state dict
    model.load_state_dict(updated_model_dict, strict=(not ignore_missing))


def find_prefix(keys: list, p_agree: float = 0.66, separator: str = ".") -> Tuple[str, int]:
    """
    Finds common prefix among state_dict keys
    :param keys: list of strings to find a common prefix
    :param p_agree: percentage of keys that should agree for a valid prefix
    :param separator: string that separates keys into substrings, e.g. "model.conv1.bias"
    :return: (prefix, end index of prefix)
    """
    keys = [k.split(separator) for k in keys]
    p_len = 0
    common_prefix = ""
    prefs = {"": len(keys)}
    while True:
        sorted_prefs = sorted(prefs.items(), key=lambda x: x[1], reverse=True)
        # check if largest count is above threshold
        if not prefs or sorted_prefs[0][1] < p_agree * len(keys):
            break
        common_prefix = sorted_prefs[0][0]  # save prefix

        p_len += 1
        prefs = {}
        for key in keys:
            if p_len == len(key):  # prefix cannot be an entire key
                continue
            p_str = ".".join(key[:p_len])
            prefs[p_str] = prefs.get(p_str, 0) + 1
    return common_prefix, p_len - 1


class Center:
    """
    Class centering readouts
    """

    def __init__(self, target_mean, mean_key="mask_mean"):
        self.target_mean = target_mean
        self.mean_key = mean_key

    def __call__(self, model):
        key_components = [
            comp for comp in zip(*[key.split(".") for key in model.state_dict().keys() if self.mean_key in key])
        ]
        mean_full_keys = [
            ".".join([key_components[j][i] for j in range(len(key_components))])
            for i, var_name in enumerate(key_components[-1])
        ]

        mod_state_dict = deepcopy(model.state_dict())
        device = mod_state_dict[mean_full_keys[0]].device
        for mean_full_key in mean_full_keys:
            mod_state_dict[mean_full_key] = torch.zeros_like(model.state_dict()[mean_full_key]) + torch.tensor(
                self.target_mean, device=device
            )
        model.load_state_dict(mod_state_dict)


def load_ensemble_retina_model_from_directory(
        directory_path: str,
        device: str = "cuda",
        center_readout: Optional[Center] = None,
) -> Tuple:
    """
    Returns an ensemble data_info object and an ensemble model that it loads from the directory path.

    The code assumes that the following files are in the directory:
    - state_dict_{seed:05d}.pth.tar
    - config_{seed:05d}.yaml
    - data_info_{seed:05d}.pkl
    where seed is an integer that represents the random seed the model was trained with
    """
    yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=SafeLoaderWithTuple)
    file_names = [f for f in os.listdir(directory_path) if f.endswith("yaml")]
    seed_array = [int(file_name[: -len(".yaml")].split("_")[1]) for file_name in file_names]
    seed_array.sort()
    model_list = []
    data_info_list = []

    for seed in seed_array:
        state_dir_path = f"{directory_path}/state_dict_{seed:05d}.pth.tar"
        model_config_path = f"{directory_path}/config_{seed:05d}.yaml"
        data_info_path = f"{directory_path}/data_info_{seed:05d}.pkl"

        state_dict = torch.load(state_dir_path)
        with open(model_config_path, "r") as f:
            config = yaml.load(f, SafeLoaderWithTuple)
        with open(data_info_path, "rb") as fb:
            data_info = pickle.load(fb)
        data_info_list.append(data_info)

        model_fn = config["model_fn"]
        repo, _, _, model_type = model_fn.split('.')
        if repo == "nnfabrik_euler":  # convert model_fn from nnfabrik to openretina
            #ToDo check robustness across model types
            model_fn = '.'.join(['openretina', 'hoefling_2024', 'models', model_type])
        elif repo == "openretina":
            pass  # nothing to change
        else:
            raise ValueError(f"Unsupported repository {repo} for loading a model. Please implement this manually.")
        model_config = config["model_config"]
        model = get_model(model_fn, model_config, seed=seed, data_info=data_info, state_dict=state_dict, strict=False)
        model_list.append(model)

    # Just put inside wrapper for ensemble
    ensemble_model = EnsembleModel(*model_list)
    # Center readouts
    if center_readout is not None:
        center_readout(ensemble_model)
    ensemble_model.to(device)
    ensemble_model.eval()

    # Based on the mei mixin code normally the first data_info entry is used, I assume they are all the same
    # See: https://github.com/eulerlab/mei/blob/d1f24ef89bdeb4643057ead2ee6b1fb651a90ebb/mei/mixins.py#L88-L94
    data_info = data_info_list[0]

    return data_info, ensemble_model


class EnsembleModel(nn.Module):
    """An ensemble model consisting of several individual ensemble members.

    Attributes:
        *members: PyTorch modules representing the members of the ensemble.
    """

    _module_container_cls = nn.ModuleList

    def __init__(self, *members: nn.Module):
        """Initializes EnsembleModel."""
        super().__init__()
        self.members = self._module_container_cls(members)

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Calculates the forward pass through the ensemble.

        The input is passed through all individual members of the ensemble and their outputs are averaged.

        Args:
            x: A tensor representing the input to the ensemble.
            *args: Additional arguments will be passed to all ensemble members.
            **kwargs: Additional keyword arguments will be passed to all ensemble members.

        Returns:
            A tensor representing the ensemble's output.
        """
        outputs = [m(x, *args, **kwargs) for m in self.members]
        mean_output = torch.stack(outputs, dim=0).mean(dim=0)
        return mean_output

    def readout_keys(self) -> list[str]:
        return self.members[0].readout_keys  # type: ignore

    def __repr__(self):
        return f"{self.__class__.__qualname__}({', '.join(m.__repr__() for m in self.members)})"
