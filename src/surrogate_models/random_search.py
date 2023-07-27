from typing import List, Tuple
import numpy as np

from src.surrogate_models.base_hyperparameter_optimizer import BaseHyperparameterOptimizer
from src.history.fidelity_manager import FidelityManager


class RandomOptimizer(BaseHyperparameterOptimizer):
    def __init__(
        self,
        hp_candidates: np.ndarray,
        fidelity_manager: FidelityManager = None,
        max_budget: int = 52,
        seed: int = 0,
        max_nr_trials=1000,
        **kwargs,
    ):
        """
        Wrapper for the Random search algorithm.

        Args:
        -----
        hyperparameter_candidates: np.ndarray
            2d array which contains all possible configurations which can be queried.
        max_budget: int
            The number of max epochs used during the HPO optimization.
        seed: int
            Seed used to reproduce the experiments.
        max_nr_trials: int
            The total runtime budget, given as the number of epochs spent during HPO.
        """
        self.hyperparameter_candidates = hp_candidates
        self.rng = np.random.RandomState(seed)
        np.random.seed(seed)
        self.evaluated_configurations = set()
        self.fidelity_manager = fidelity_manager
        self.max_budget = max_budget
        self.max_trials = max_nr_trials
        self.extra_args = kwargs
        self.previous_fidelity = None
        self.previous_config_index = None

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
        while True:
            if self.previous_fidelity is None:
                possible_candidates = {i for i in range(self.hyperparameter_candidates.shape[0])}
                not_evaluated_candidates = possible_candidates - self.evaluated_configurations
                config_index = np.random.choice(list(not_evaluated_candidates))
                self.evaluated_configurations.add(config_index)
                self.previous_config_index = config_index

            current_config_id = self.previous_config_index

            fidelity = self.fidelity_manager.get_next_fidelity_id(configuration_id=current_config_id)
            self.previous_fidelity = fidelity

            if fidelity is not None:
                self.fidelity_manager.set_fidelity_id(configuration_id=current_config_id, fidelity_id=fidelity)
                break

        return [current_config_id], [fidelity]

    def observe(self, hp_index: int, budget: List[Tuple[int]], hp_curve: List[float]):
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
        pass
