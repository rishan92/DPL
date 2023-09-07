import numpy as np
from typing import List, Union, Dict, Tuple, Optional
import ConfigSpace as CS
import itertools
from collections import OrderedDict
from ConfigSpace.hyperparameters import FloatHyperparameter, IntegerHyperparameter, Constant, \
    CategoricalHyperparameter, OrdinalHyperparameter


# from numpy.typing import NDArray


class FidelityManager:
    def __init__(self, fidelity_space: CS.ConfigurationSpace, num_configurations: int, max_steps: int):
        self.max_steps = max_steps
        self.raw_fidelity_space = None
        self.fidelity_interval = {}
        self.raw_fidelity_space = fidelity_space
        self.config_space_info = self.extract_hyperparameter_info(fidelity_space)
        fidelity_space = self.from_config_space(self.config_space_info)
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
        self.first_fidelity = self.convert_fidelity_id_to_fidelity(fidelity_id=self.first_fidelity_id)
        self.last_fidelity = self.convert_fidelity_id_to_fidelity(fidelity_id=self.last_fidelity_id)
        self.last_fidelity = self.convert_type(fidelity=self.last_fidelity)
        self.predict_fidelity_mode = "all"  # "next"  #
        self.predict_fidelities = self.generate_predict_fidelity_tensor()
        self.evaluated_fidelity_id_map = [set()] * self.num_configurations

    def get_raw_fidelity_space(self):
        return self.raw_fidelity_space

    def get_max_fidelity(self):
        return {k: v for k, v in zip(self.fidelity_names, self.last_fidelity)}

    def get_min_fidelity(self):
        return {k: v for k, v in zip(self.fidelity_names, self.first_fidelity)}

    def get_predict_fidelity_ids(self):
        return self.predict_fidelities

    def from_config_space(self, config_space):
        fidelity_curve_points = {}
        for k, v in config_space.items():
            if v[2] == 'ord':
                fidelity_curve_points[k] = np.array(v[4])
            else:
                fidelity_curve_points[k] = np.around(
                    np.linspace(v[0], v[1], self.max_steps), decimals=4
                )
                if v[2] == 'int':
                    rounded_values = np.round(fidelity_curve_points[k]).astype(int)
                    curve_values = np.unique(rounded_values)
                    curve_values = curve_values.astype(float)
                    fidelity_curve_points[k] = curve_values
                elif v[2] == 'float':
                    interval = (v[1] - v[0]) / self.max_steps
                else:
                    raise NotImplementedError

        return fidelity_curve_points

    def convert_fidelity_id_to_fidelity(self, fidelity_id, is_normalized=False):
        fidelity_map = \
            self.fidelity_id_to_normalized_fidelity_map if is_normalized else self.fidelity_id_to_fidelity_map
        return fidelity_map[fidelity_id]

    def convert_fidelity_to_fidelity_id(self, fidelity):
        return self.fidelity_to_fidelity_id_map[fidelity]

    def convert_type(self, fidelity):
        if isinstance(fidelity, Dict):
            converted_fidelity = {}
            for k, v in fidelity.items():
                if self.config_space_info[k][2] == 'int':
                    converted_fidelity[k] = int(v)
                else:
                    converted_fidelity[k] = v
        else:
            converted_fidelity = []
            for k, v in zip(self.fidelity_names, fidelity):
                if self.config_space_info[k][2] == 'int':
                    converted_fidelity.append(int(v))
                else:
                    converted_fidelity.append(v)
            converted_fidelity = tuple(converted_fidelity)

        return converted_fidelity

    def normalize_fidelity(self, fidelity):
        normalized_fidelity = tuple([f / max_f for f, max_f in zip(fidelity, self.last_fidelity)])
        return normalized_fidelity

    def generate_predict_fidelity_tensor(self):
        for k, v in self.fidelity_space.items():
            assert np.all(np.diff(v) > 0), "Fidelities should be strictly increasing."
            self.fidelity_space[k] = np.array(v)

        combinations = list(itertools.product(*self.fidelity_space.values()))
        fidelities = [
            (tuple(np.where(self.fidelity_space[key] == value)[0][0]
                   for key, value in zip(self.fidelity_space, combination)), combination)
            for combination in combinations
        ]
        return fidelities

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

    def get_next_fidelity_id(self, configuration_id, configuration_fidelity_id=None):
        if self.predict_fidelity_mode == "all":
            if configuration_fidelity_id is not None:
                return configuration_fidelity_id
            else:
                raise NotImplementedError
        elif self.predict_fidelity_mode == "next":
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
        else:
            raise NotImplementedError

    def get_fidelity_id(self, configuration_id):
        return self.fidelity_ids[configuration_id]

    def get_next_fidelity(self, configuration_id, is_normalized=False, configuration_fidelity=None):
        next_id = self.get_next_fidelity_id(
            configuration_id=configuration_id, configuration_fidelity_id=configuration_fidelity
        )
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
        self.evaluated_fidelity_id_map[configuration_id].add(fidelity_id)

    def add_fidelity(self, configuration_id):
        assert configuration_id == len(self.fidelity_ids), "Only incremental adding of fidelity supported."
        self.fidelity_ids.append(None)

    def extract_hyperparameter_info(self, config_space: CS.ConfigurationSpace):
        hyperparameter_info = OrderedDict()
        for hp in list(config_space.values()):
            hp_name = hp.name
            default_value = hp.default_value
            is_log = False
            categories = []
            if isinstance(hp, Constant):
                value = hp.value
                if isinstance(value, float):
                    hp_type = 'float'
                elif isinstance(value, int):
                    hp_type = 'int'
                elif isinstance(value, str):
                    hp_type = 'str'
                    categories = [value]
                else:
                    raise NotImplementedError
                lower = upper = value
            elif isinstance(hp, FloatHyperparameter):
                hp_type = 'float'
                is_log = hp.log
                lower = hp.lower
                upper = hp.upper
            elif isinstance(hp, IntegerHyperparameter):
                hp_type = 'int'
                is_log = hp.log
                lower = hp.lower
                upper = hp.upper
            elif isinstance(hp, CategoricalHyperparameter):
                hp_type = 'str'
                lower = 0
                upper = 0
                categories = hp.choices
            elif isinstance(hp, OrdinalHyperparameter):
                hp_type = 'ord'
                lower = 0
                upper = len(hp.sequence) - 1
                categories = hp.sequence
            else:
                raise NotImplementedError(f"Hyperparameter type not implemented: {hp}")

            hyperparameter_info[hp_name] = [lower, upper, hp_type, is_log, categories, default_value]

        return hyperparameter_info
