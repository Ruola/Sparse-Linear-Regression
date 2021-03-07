import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os

from algorithms.gradient_descent import GradientDescent
from algorithms.iterative_threshold_methods import IterativeThresholdMethods
import utils.constants as constants
from utils.draw import Draw
from utils.error import Error
from utils.generate_data import GenerateData


class ChangeConditionNumber:
    """Simulations on the change of condition number of design matrix.
    """
    def __init__(self):
        """Set the type of design matrix, anisotropic and some constants.
        """
        self.design = constants.ANISOTROPIC_NAME
        self.steps = constants.GD_STEPS  # number of experiments
        self.num_iter = constants.GD_NUM_ITERATION  # number of iterations
        self.x_original = constants.X
        self.gd_types = (constants.GD_NAME, constants.NEWTON_NAME)
        self.iter_types = (constants.IHT_NAME, )
        self.iterative_threshold_methods = (constants.ISTA_NAME, )

    def _update_algos_map(self, algos_map, algo_name, kappa, error_added):
        """A private function to update @param algos_map.
        
        @param algos_map: key is algo name, value is a dict with kappa-error as key-value pair.
        @param algo_name: the key of algos_map which needs an update.
            e.g. constants.ISTA_NAME, constants.GD_NAME+constants.IHT_NAME, see utils.constants.
        @param kappa: (condition number) It is a key of the map, algos_map[algo_name].
        @param error_added: the error of an experiment with kappa as condition numebr
            and algo_name as the algorithm.
        @return the updated map.
        """
        if algos_map.get(algo_name):
            curr_kappa_error_map = algos_map[algo_name]
        else:
            curr_kappa_error_map = dict()
        if curr_kappa_error_map.get(kappa):
            # if curr_kappa_error_map has a key, kappa.
            curr_kappa_error_map[kappa] += error_added
        else:  # if curr_kappa_error_map never see this key.
            curr_kappa_error_map[kappa] = error_added
        algos_map[algo_name] = curr_kappa_error_map
        return algos_map

    def run_one_experiment(self, kappa):
        """Run one experient working for multiprocessing.
        
        @param kappa - condition number.
        @return algo_name gener_error hashmap.
        """
        y, H, self.SIGMA_half = GenerateData(self.design,
                                             kappa).generate_data()
        algos_map = dict()
        for algo_name in self.iterative_threshold_methods:
            _, _, gener_error = IterativeThresholdMethods().get_errors_by_cv(
                self.x_original, y, H, self.num_iter, self.SIGMA_half,
                algo_name, False)
            # To add {algo_name: {kappa, final_error}} key-value-pair to algos_map.
            algos_map = self._update_algos_map(algos_map, algo_name, kappa,
                                               gener_error[-1])
            print(algo_name, " error ", gener_error[-1])
        for gd_type in self.gd_types:
            for iter_type in self.iter_types:
                _, _, gener_error = GradientDescent().get_errors_by_cv(
                    self.x_original, y, H, self.num_iter, self.SIGMA_half,
                    gd_type, iter_type, False)
                algo_name = gd_type + iter_type
                algos_map = self._update_algos_map(algos_map, algo_name, kappa,
                                                   gener_error[-1])
                print(algo_name, " error ", gener_error[-1])
        return algos_map

    def compare_condition_numbers(self):
        """Get the change of generalization error w.r.t. condition number.
        """
        algos_map = {}
        for _ in range(self.steps):  # Run several experiments
            pool = mp.Pool(mp.cpu_count())
            pool_results = pool.map(self.run_one_experiment, np.arange(1, 40, 5), 1)
            for map_result in pool_results:
                for algo_name in map_result:
                    for kappa in map_result[algo_name]:
                        algos_map = self._update_algos_map(
                            algos_map, algo_name, kappa,
                            map_result[algo_name][kappa])

        # To take average on the error of experiments.
        for algo_name in algos_map:
            curr_kappa_error_map = algos_map[algo_name]
            for kappa in curr_kappa_error_map:  # to update curr_kappa_error_map
                total_error = curr_kappa_error_map[
                    kappa]  # Total errors of all experiments.
                curr_kappa_error_map[
                    kappa] = total_error / self.steps  # To take average.
            algos_map[algo_name] = curr_kappa_error_map  # to update algos_map
        for algo_name in algos_map:
            Draw().plot_using_a_map(algos_map[algo_name], algo_name)
        plt.xlabel("condition number")
        plt.ylabel("generalization error")
        plt.legend()
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) +
            "/figures/condition number/comparison by change of condition number0"
        )
        plt.clf()


if __name__ == "__main__":
    ChangeConditionNumber().compare_condition_numbers()
