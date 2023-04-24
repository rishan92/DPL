import os
from importlib import import_module
from typing import List, Tuple, Dict, Optional, Any, Union, Type
import shutil


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


def get_class_from_package(package, name):
    return getattr(package, name)


def get_class_from_packages(packages: List, name):
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


def merge_dicts(dict1, dict2):
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


def delete_folder_content(folder_path: str):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        if os.path.isfile(item_path):
            os.remove(item_path)  # Remove file
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
