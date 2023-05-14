import os
from importlib import import_module
from typing import List, Tuple, Dict, Optional, Any, Union, Type
import shutil
from pathlib import Path
import torch
import warnings
import numpy as np
from scipy.stats import norm

import src.models.activation_functions


def get_class(folder_path, class_name):
    files = os.listdir(folder_path)

    for file in files:
        if file.endswith(".py") and not file.startswith("__"):
            # Remove the .py extension
            module_name = file[:-3]
            module_path = f"{folder_path.replace('/', '.')}.{module_name}"
            module = import_module(module_path)

            class_obj = getattr(module, class_name, None)
            if class_obj:
                return class_obj

    return None


def get_class_from_package(package, name: str):
    return getattr(package, name)


def get_class_from_packages(packages: List, name: str):
    attribute = None
    for package in packages:
        try:
            attribute = getattr(package, name)
            break
        except AttributeError:
            continue

    if attribute is None:
        raise AttributeError(f"Attribute '{name}' not found in any of the given packages.")
    return attribute


def merge_dicts(dict1: Dict, dict2: Dict):
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


# Custom decorator to define a property as a class property

class classproperty(property):
    def __get__(self, *args, **kwargs):
        # If no owner is provided, use None
        owner = args[1] if len(args) > 1 else None
        return classmethod(self.fget).__get__(None, owner)()


def delete_folder_content(folder_path: Path):
    for item in os.listdir(folder_path):
        item_path = folder_path / str(item)

        if item_path.is_file():
            item_path.unlink()  # Delete file
        elif item_path.is_dir():
            shutil.rmtree(item_path)


def numpy_to_torch_apply(numpy_array, torch_function):
    tensor = torch.from_numpy(numpy_array)
    result_tensor = torch_function(tensor)
    return result_tensor.numpy()


def get_inverse_function_class(act_func_name: str):
    inverse_class = None
    if act_func_name:
        if act_func_name == "Sigmoid":
            inverse_function = "InverseSigmoid"
        elif act_func_name == "ClipSigmoid":
            inverse_function = "InverseClipSigmoid"
        elif act_func_name == "ClipLeakyReLU":
            inverse_function = "InverseClipLeakyReLU"
        elif act_func_name == "OffsetTanh":
            inverse_function = "InverseOffsetTanh"
        elif act_func_name == "Identity":
            inverse_function = "Identity"
        elif act_func_name in ["ClipReLU"]:
            inverse_function = "Identity"
            warnings.warn(
                f"Using output_act_func {act_func_name} would produce "
                f"non power law curves as outputs."
            )
        else:
            raise NotImplementedError(
                f"Using output_act_func {act_func_name} is not supported."
            )

        inverse_class = get_class_from_packages([torch.nn, src.models.activation_functions], inverse_function)
    return inverse_class


def acq(
    best_values: np.ndarray,
    mean_predictions: np.ndarray,
    std_predictions: np.ndarray,
    explore_factor: float = 0.25,
    acq_mode: str = 'ei',
) -> np.ndarray:
    if acq_mode == 'ei':
        difference = np.subtract(best_values, mean_predictions)

        zero_std_indicator = np.zeros_like(std_predictions, dtype=bool)
        zero_std_indicator[std_predictions == 0] = True
        not_zero_std_indicator = np.invert(zero_std_indicator)
        z = np.divide(difference, std_predictions, where=not_zero_std_indicator)
        z[zero_std_indicator] = 0

        acq_values = np.add(np.multiply(difference, norm.cdf(z)), np.multiply(std_predictions, norm.pdf(z)))
    elif acq_mode == 'ucb':
        # we are working with error rates so we multiply the mean with -1
        acq_values = np.add(-1 * mean_predictions, explore_factor * std_predictions)
    elif acq_mode == 'thompson':
        acq_values = np.random.normal(mean_predictions, std_predictions)
    elif acq_mode == 'exploit':
        acq_values = mean_predictions
    else:
        raise NotImplementedError(
            f'Acquisition function {acq_mode} has not been'
            f'implemented',
        )

    return acq_values
