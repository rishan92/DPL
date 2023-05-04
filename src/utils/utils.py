import os
from importlib import import_module
from typing import List, Tuple, Dict, Optional, Any, Union, Type
import shutil
from pathlib import Path


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
