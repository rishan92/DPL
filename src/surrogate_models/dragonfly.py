import time
from argparse import Namespace
import threading
import numpy as np
from typing import Dict, List, OrderedDict, Tuple
import ConfigSpace as CS
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, OrdinalHyperparameter

from dragonfly.dragonfly import load_config, maximize_multifidelity_function, minimize_multifidelity_function

from src.surrogate_models.base_hyperparameter_optimizer import BaseHyperparameterOptimizer
from src.history.fidelity_manager import FidelityManager


class DragonFlyOptimizer(BaseHyperparameterOptimizer):
    def __init__(
        self,
        hp_candidates: np.ndarray,
        fidelity_manager: FidelityManager = None,
        seed: int = 0,
        max_budget: int = 52,
        total_budget: int = 1000,
        maximization: bool = True,
        **kwargs,
    ):
        """
        Wrapper for the BOCA algorithm.

        Args:
        -----
        hyperparameter_candidates: np.ndarray
            2d array which contains all possible configurations which can be queried.
        param_space: OrderedDict
            The hyperparameter search-space, indicating the type and range of every
            hyperparameter.
        seed: int
            Seed used to reproduce the experiments.
        max_budget: int
            The number of maximal steps for a hyperparameter configuration.
        max_nr_trials: int
            The total runtime budget, given as the number of epochs spent during HPO.
        maximization: bool
            If the inner objective is to maximize or minimize.
        """
        self.maximization = maximization
        self.hyperparameter_candidates = hp_candidates
        self.fidelity_manager = fidelity_manager
        self.extra_arguments = kwargs

        self.hyperparameter_mapping = self.create_configuration_to_indices()

        # empty configuration, empty budget, empty information for config
        self.next_conf = None
        self.conf_budget = None
        self.conf_info = None
        self.fidelity_index = None
        self.rng = np.random.RandomState(seed)
        np.random.seed(seed)

        self.evaluated_configurations = dict()
        self.evaluated_hp_curves = dict()
        self.evaluated_budgets = dict()
        self.to_add_evaluated_budgets = dict()
        # Basically the same as evaluated_hp_curves. However, this will
        # be used to estimate the evaluation cost for a certain fidelity.
        # If we used evaluated_hp_curves, the cost would always be zero
        # since the configuration index is added there as evaluated already
        # before.
        self.fidelity_hp_curves = dict()
        domain_vars = [
            {'type': 'discrete_euclidean', 'items': list(self.hyperparameter_candidates)},
        ]

        # fidel_vars = [
        #     {'type': 'discrete_numeric', 'items': fidelity_space['f0']},
        #     {'type': 'discrete_numeric', 'items': fidelity_space['f1']},
        # ]
        # fidel_vars = [
        #     {'type': 'int', 'min': 10, 'max': 5000},
        #     {'type': 'float', 'min': 1, 'max': 10},
        # ]
        fidelity_space_cs = self.fidelity_manager.get_raw_fidelity_space()
        fidel_vars = self.convert_to_dragonfly_space(fidelity_space_cs)

        fidel_to_opt = self.fidelity_manager.last_fidelity

        config = {
            'domain': domain_vars,
            'fidel_space': fidel_vars,
            'fidel_to_opt': fidel_to_opt,
        }
        # How frequently to build a new (GP) model
        # --build_new_model_every 17
        options_namespace = Namespace(
            gpb_hp_tune_criterion='ml',
        )
        config = load_config(config)

        self.previous_fidelity = None
        self.previous_config_index = None

        self.dragonfly_run = threading.Thread(
            target=maximize_multifidelity_function if self.maximization else minimize_multifidelity_function,
            kwargs={
                'func': self.target_function,
                'max_capital': total_budget,
                'config': config,
                'domain': config.domain,
                'fidel_space': config.fidel_space,
                'fidel_to_opt': config.fidel_to_opt,
                'options': options_namespace,
                'fidel_cost_func': self.fidel_cost_function,
            },
            daemon=True,
        )
        self.dragonfly_run.start()

    def fidel_cost_function(self, fidelity, configuration):

        fidelity = tuple(fidelity)
        normalized_fidelity = self.fidelity_manager.normalize_fidelity(fidelity=fidelity)

        if configuration is None:
            fidelity_opt_cost = sum(normalized_fidelity)
            fidelity_opt_cost /= len(fidelity)
            print("fidel_cost_function_No_Config", fidelity, fidelity_opt_cost)
            return fidelity_opt_cost

        configuration_id = self.map_configuration_to_index(list(configuration[0]))

        if configuration_id in self.evaluated_budgets:
            previous_budgets = self.evaluated_budgets[configuration_id]
            max_previous_fidelity = [max(element) for element in zip(*previous_budgets)]
        else:
            max_previous_fidelity = [0] * len(fidelity)

        max_previous_normalized_fidelity = self.fidelity_manager.normalize_fidelity(fidelity=max_previous_fidelity)

        # budget_cost = budget - previous_budget
        fidelity_opt_cost = sum(max(a - b, 0) for a, b in zip(normalized_fidelity, max_previous_normalized_fidelity))
        fidelity_opt_cost /= len(fidelity)

        # add to evaluated_budgets. This is called here not in the observe function, because dragonfly calls
        # target function first and then the fidelity cost function to get related cost. Also it sometimes only calls
        # fidelity cost function without calling target function.
        if configuration_id in self.to_add_evaluated_budgets:
            if fidelity in self.to_add_evaluated_budgets[configuration_id]:
                if configuration_id not in self.evaluated_budgets:
                    self.evaluated_budgets[configuration_id] = []
                self.evaluated_budgets[configuration_id].append(fidelity)

                self.to_add_evaluated_budgets[configuration_id].remove(fidelity)
                if len(self.to_add_evaluated_budgets[configuration_id]) == 0:
                    self.to_add_evaluated_budgets.pop(configuration_id)

        print("fidel_cost_function", fidelity, fidelity_opt_cost, configuration_id)
        return fidelity_opt_cost

    def convert_to_dragonfly_space(self, config_space):
        fidel_vars = []
        for hp in config_space.get_hyperparameters():
            var = {}
            if isinstance(hp, UniformFloatHyperparameter):
                var['type'] = 'float'
                var['min'] = hp.lower
                var['max'] = hp.upper
            elif isinstance(hp, UniformIntegerHyperparameter):
                var['type'] = 'int'
                var['min'] = hp.lower
                var['max'] = hp.upper
            elif isinstance(hp, CategoricalHyperparameter):
                var['type'] = 'discrete'
                var['items'] = hp.choices
            elif isinstance(hp, OrdinalHyperparameter):
                var['type'] = 'discrete_numeric'
                var['items'] = hp.sequence
            else:
                raise NotImplementedError

            fidel_vars.append(var)

        return fidel_vars

    def target_function(
        self,
        budget: List[int],
        config: List[np.ndarray],
    ) -> float:
        """
        Function to evaluate for a given configuration.

        Args:
        -----
        budget: list
            The budget for which the configuration will be run.
        config: list
            Configuration suggested by DragonFly.

        Returns:
        ________
        score: float
            A score which indicates the validation performance
            of the configuration.
        """
        # the budget is a list initially with only one value
        budget = tuple(budget)
        # if budget is not None:
        #     budget = int(budget)

        # initially the config is a list consisting of a single np.ndarray
        config = list(config[0])

        config_index = self.map_configuration_to_index(config)

        # not the first hyperparameter to be evaluated for the selected
        # budget
        if budget in self.evaluated_configurations:
            self.evaluated_configurations[budget].add(config_index)
        else:
            self.evaluated_configurations[budget] = set([config_index])

        self.conf_budget = budget

        need_to_query_framework = True
        if config_index in self.evaluated_hp_curves:
            config_curve = self.evaluated_hp_curves[config_index]
            # the hyperparameter configuration has been evaluated before
            # and it was evaluated for a higher\same budget
            if budget in config_curve:
                need_to_query_framework = False

        # Save the config index in fidelity index, since sometimes this config
        # if evaluated before it is not passed to the framework, but it would be
        # still needed for the cost estimation.
        self.fidelity_index = config_index

        if need_to_query_framework:
            # update the field so the framework can take the index and
            # reply
            self.next_conf = config_index
            while True:
                if self.conf_info is not None:
                    score = self.conf_info['score']
                    # val_curve = self.conf_info['val_curve']
                    # # save the curve for the evaluated hyperparameter
                    # # configuration
                    # if config_index not in self.evaluated_hp_curves:
                    #     self.evaluated_hp_curves[config_index] = {}
                    # self.evaluated_hp_curves[config_index][budget] = score
                    break
                else:
                    # The framework has not yet responded with a value,
                    # keep checking
                    # TODO add a delay
                    time.sleep(1)
        else:
            score = config_curve[budget]

        # need to make the previous response None since DragonFly
        # continues running in the background
        self.conf_info = None
        print("target_function", config_index, budget, score)
        return score

    def suggest(self) -> Tuple[int, int]:
        """
        Get information about the next configuration.

        Returns:
        ________
        next_conf, conf_budget: tuple
            A tuple that contains information about the next
            configuration (index in the hyperparameter_candidates it was
            given) and the budget for the hyperparameter to be evaluated
            on.
        """
        # if self.previous_fidelity is None:
        #     while self.next_conf is None:
        #         # DragonFly has not generated the config yet
        #         time.sleep(1)
        #     self.conf_info = None
        #     self.previous_config_index = self.next_conf
        #
        # req_budget_id = self.fidelity_manager.convert_fidelity_to_fidelity_id(self.conf_budget)
        #
        # current_config_id = self.previous_config_index
        #
        # fidelity_id = self.fidelity_manager.get_next_fidelity_id(configuration_id=current_config_id)
        # if fidelity_id != req_budget_id:
        #     self.previous_fidelity_id = fidelity_id
        # else:
        #     self.previous_fidelity_id = None
        #
        # if fidelity_id is not None:
        #     self.fidelity_manager.set_fidelity_id(configuration_id=current_config_id, fidelity_id=fidelity_id)

        while self.next_conf is None:
            # DragonFly has not generated the config yet
            time.sleep(1)
        self.conf_info = None
        # fidelity_id = self.fidelity_manager.convert_fidelity_to_fidelity_id(self.conf_budget)
        print("suggest", self.next_conf, self.conf_budget)
        return [self.next_conf], [self.conf_budget]

    def observe(
        self,
        hp_index: int,
        budget: int,
        hp_curve: List[float],
    ):
        """
        Respond regarding the performance of a
        hyperparameter configuration. get_next should
        be called first to retrieve the configuration.

        Args:
        -----
        hp_index: int
            The index of the evaluated hyperparameter configuration.
        budget: int
            The budget for which the hyperparameter configuration was evaluated.
        hp_curve: np.ndarray, list
            validation accuracy curve. The last value is the same as the score.
        """
        # req_budget_id = self.fidelity_manager.convert_fidelity_to_fidelity_id(self.conf_budget)
        # converted_budget = self.fidelity_manager.convert_fidelity_id_to_fidelity(budget[-1])
        if hp_index not in self.evaluated_hp_curves:
            self.evaluated_hp_curves[hp_index] = {}
        self.evaluated_hp_curves[hp_index][budget[-1]] = hp_curve[-1]

        if hp_index not in self.to_add_evaluated_budgets:
            self.to_add_evaluated_budgets[hp_index] = []
        self.to_add_evaluated_budgets[hp_index].append(budget[-1])

        #
        # if budget[-1] == req_budget_id and hp_index == self.previous_config_index:
        #     assert self.next_conf is not None, 'Call get_next first.'
        #     self.next_conf = None
        #
        #     self.conf_info = {
        #         'score': hp_curve[-1],
        #         'val_curve': hp_curve,
        #         'fidelity': budget,
        #     }
        assert self.next_conf is not None, 'Call get_next first.'
        self.next_conf = None

        self.conf_info = {
            'score': hp_curve[-1],
            'val_curve': hp_curve,
            'fidelity': budget,
        }

    def create_configuration_to_indices(
        self,
    ) -> Dict[Tuple, int]:
        """
        Maps every configuration to its index as specified
        in hyperparameter_candidates.

        Args:
        -----
        hyperparameter_candidates: np.ndarray
            All the possible hyperparameter candidates given
            by the calling framework.

        Returns:
        ________
        hyperparameter_mapping: dict
            A dictionary where the keys are tuples representing
            hyperparameter configurations and the values are indices
            representing their placement in hyperparameter_candidates.
        """
        hyperparameter_mapping = dict()
        for i in range(0, self.hyperparameter_candidates.shape[0]):
            hyperparameter_mapping[tuple(self.hyperparameter_candidates[i])] = i

        return hyperparameter_mapping

    def map_configuration_to_index(
        self,
        hyperparameter_candidate: List,
    ) -> int:
        """
        Return the index of the hyperparameter_candidate from
        the given initial array of possible hyperparameters.

        Args:
        -----
        hyperparameter_candidate: np.ndarray
            Hyperparameter configuration.

        Returns:
        ________
        index of the hyperparameter_candidate.
        """
        hyperparameter_candidate = tuple(hyperparameter_candidate)

        return self.hyperparameter_mapping[hyperparameter_candidate]
