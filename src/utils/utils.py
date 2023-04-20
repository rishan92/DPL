import os
from importlib import import_module


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


