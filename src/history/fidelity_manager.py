import numpy as np
from typing import List, Union, Dict, Tuple, Optional
import ConfigSpace as CS
import itertools


# from numpy.typing import NDArray


class FidelityManager:
    def __init__(self, fidelity_space: Dict[str, Union[List, np.ndarray]], num_configurations: int):
        self.fidelity_space = fidelity_space
        self.fidelity_names = list(self.fidelity_space.keys())
        self.num_configurations = num_configurations
        self.fidelity_id_to_fidelity_map, self.fidelity_id_to_normalized_fidelity_map = self.generate_fidelity_tensor()
        self.fidelity_to_fidelity_id_map = self.get_reverse_mapping(self.fidelity_id_to_fidelity_map)
        self.fidelity_ids: List[Optional[Tuple[int]]] = [None] * self.num_configurations
        self.fidelity: List[Optional[Tuple[int]]] = [None] * self.num_configurations
        self.fidelity_path_ids: List[Tuple[int]] = []
        self.first_fidelity_id: Tuple[int] = tuple([0] * len(self.fidelity_space))
        self.last_fidelity_id: Tuple[int] = tuple([len(self.fidelity_space[k]) - 1 for k in self.fidelity_names])

    def convert_fidelity_id_to_fidelity(self, fidelity_id, is_normalized=False):
        fidelity_map = \
            self.fidelity_id_to_normalized_fidelity_map if is_normalized else self.fidelity_id_to_fidelity_map
        return fidelity_map[fidelity_id]

    def convert_fidelity_to_fidelity_id(self, fidelity):
        return self.fidelity_to_fidelity_id_map[fidelity]

    def min_fidelity(self):
        return self.fidelity_id_to_fidelity_map[self.first_fidelity_id]

    def get_max_fidelity(self):
        return self.fidelity_id_to_fidelity_map[self.last_fidelity_id]

    def generate_fidelity_tensor(self):
        for k, v in self.fidelity_space.items():
            assert np.all(np.diff(v) > 0), "Fidelities should be strictly increasing."
            self.fidelity_space[k] = np.array(v)

        combinations = list(itertools.product(*self.fidelity_space.values()))
        fidelities = {
            tuple(np.where(self.fidelity_space[key] == value)[0][0]
                  for key, value in zip(self.fidelity_space, combination)): combination
            for combination in combinations
        }

        normalized_fidelity_space = {}
        for k, v in self.fidelity_space.items():
            normalized_fidelity_space[k] = v / v[-1]
        combinations = list(itertools.product(*normalized_fidelity_space.values()))
        normalized_fidelities = {
            tuple(np.where(normalized_fidelity_space[key] == value)[0][0]
                  for key, value in zip(normalized_fidelity_space, combination)): combination
            for combination in combinations
        }

        return fidelities, normalized_fidelities

    def get_reverse_mapping(self, mapping):
        reverse_mapping = {value: key for key, value in mapping.items()}
        return reverse_mapping

    def get_next_fidelity_id(self, configuration_id):
        num_max_budgets = 0
        next_budget = []
        fidelity_id = self.fidelity_ids[configuration_id]

        if fidelity_id is None:
            return self.first_fidelity_id

        for i, k in enumerate(self.fidelity_names):
            next_b = fidelity_id[i] + 1
            if next_b >= len(self.fidelity_space[k]):
                next_b = fidelity_id[i]
                num_max_budgets += 1
            next_budget.append(next_b)
        if num_max_budgets == len(self.fidelity_names):
            return None

        return tuple(next_budget)

    def get_fidelity_id(self, configuration_id):
        return self.fidelity_ids[configuration_id]

    def get_next_fidelity(self, configuration_id, is_normalized=False):
        next_id = self.get_next_fidelity_id(configuration_id=configuration_id)
        next_fidelity = None
        if next_id is not None:
            fidelity_map = \
                self.fidelity_id_to_normalized_fidelity_map if is_normalized else self.fidelity_id_to_fidelity_map
            next_fidelity = fidelity_map[next_id]

        return next_fidelity

    def get_fidelities(self, fidelity_ids, is_normalized=False, return_dict=False):
        is_list_input = isinstance(fidelity_ids, list)
        if not is_list_input:
            fidelity_ids = [fidelity_ids]

        fidelity_map = \
            self.fidelity_id_to_normalized_fidelity_map if is_normalized else self.fidelity_id_to_fidelity_map

        fidelities = [fidelity_map[f_id] for f_id in fidelity_ids]

        if return_dict:
            fidelities = [dict(zip(self.fidelity_names, fidelity)) for fidelity in fidelities]

        return fidelities if is_list_input else fidelities[0]

    # def get_fidelities(self, configuration_ids, is_normalized=False):
    #     fidelities = []
    #     fidelity_map = \
    #         self.fidelity_id_to_normalized_fidelity_map if is_normalized else self.fidelity_id_to_fidelity_map
    #     for i in configuration_ids:
    #         fidelity_id = self.fidelity_ids[i]
    #         if fidelity_id is None:
    #             continue
    #         fidelity = fidelity_map[fidelity_id]
    #         fidelities.append(fidelity)
    #
    #     return fidelities

    # def increment_fidelity(self, configuration_id):
    #     if self.fidelity_ids[configuration_id] is None:
    #         self.fidelity_ids[configuration_id] = self.first_fidelity_id
    #     else:
    #         self.fidelity_ids[configuration_id] += 1

    def set_fidelity_id(self, configuration_id, fidelity_id):
        self.fidelity_ids[configuration_id] = fidelity_id

    def add_fidelity(self, configuration_id):
        assert configuration_id == len(self.fidelity_ids), "Only incremental adding of fidelity supported."
        self.fidelity_ids.append(None)
