from typing import Dict, Any
from types import SimpleNamespace
from abc import ABC, abstractmethod


class Meta(ABC):
    meta = None

    def get_meta(self, **kwargs):
        return vars(self.meta)

    @staticmethod
    @abstractmethod
    def get_default_meta(**kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def set_meta(cls, config=None, **kwargs):
        config = {} if config is None else config
        default_meta = cls.get_default_meta()
        meta = {**default_meta, **config}
        cls.meta = SimpleNamespace(**meta)
        return meta
