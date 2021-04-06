import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import multiprocessing as mp
import numpy as np
import os

from algorithms.gradient_descent import GradientDescent
from algorithms.iterative_threshold_methods import IterativeThresholdMethods
import utils.constants as constants
from utils.draw import Draw
from utils.error import Error
from utils.generate_data import GenerateData


class ChangeSignal:
    """Simulations on the change of exact recovery w.r.t. the signal.
    """
    def __init__(self, kappa, error_name=constants.EXACT_RECOVERY_NAME):
        """Initialize.
        """
        self.kappa = kappa
        self.error_name = error_name
        self.design = constants.ANISOTROPIC_NAME
        self.steps = constants.GD_STEPS  # number of experiments
        self.num_iter = constants.GD_NUM_ITERATION  # number of iterations
        self.gd_types = (constants.GD_NAME, constants.FAST_NEWTON_NAME)
        self.iter_types = (constants.IHT_NAME, constants.HTP_NAME)
        self.iterative_threshold_methods = (constants.ISTA_NAME, )

    def _update_algos_map(self, algos_map, algo_name, a, error_added):
        """A private function to update @param algos_map.
        
        @param algos_map: key is algo name, value is a dict with signal-error as key-value pair.
        @param algo_name: the key of algos_map which needs an update.
            e.g. constants.ISTA_NAME, constants.GD_NAME+constants.IHT_NAME, see utils.constants.
        @param a - the signal value.
        @param error_added: the error to be added to the map.
        @return the updated map.
        """
        if algos_map.get(algo_name):
            curr_map = algos_map[algo_name]
        else:
            curr_map = dict()
        if curr_map.get(a):
            curr_map[a] += error_added
        else:
            curr_map[a] = error_added
        algos_map[algo_name] = curr_map
        return algos_map

    def run_one_experiment(self, a):
        """Run one experiment on the change of exact recovery w.r.t. the signal.
        
        @param a - the value of the true signal
        @return algo_name gener_error hashmap.
        """
        signal = a * np.ones((constants.P))
        signal[constants.S:] = 0
        signal = np.random.permutation(signal)
        y, H, self.SIGMA_half = GenerateData(self.design, self.kappa,
                                             signal).generate_data()
        algos_map = dict()
        for algo_name in self.iterative_threshold_methods:
            _, best_lambda, gener_error = IterativeThresholdMethods(
                self.error_name).get_errors_by_cv(signal, y, H, self.num_iter,
                                                  self.SIGMA_half, algo_name,
                                                  False)
            # To add {algo_name: {a, final_error}} key-value-pair to algos_map.
            algos_map = self._update_algos_map(algos_map, algo_name, a,
                                               gener_error[-1])
            print(algo_name, best_lambda)
        for gd_type in self.gd_types:
            for iter_type in self.iter_types:
                _, best_lambda, gener_error = GradientDescent(
                ).get_errors_by_cv(signal, y, H, self.num_iter,
                                   self.SIGMA_half, gd_type, iter_type, False)
                algo_name = gd_type + "+" + iter_type
                algos_map = self._update_algos_map(algos_map, algo_name, a,
                                                   gener_error[-1])
                print(algo_name, best_lambda)
        return algos_map

    def change_signal(self):
        """Run several experiments and get the change of exact recovery w.r.t. signal.
        """
        algos_map = {}
        for _ in range(self.steps):  # Run several experiments
            for a in [0.001, 0.005, 0.01, 0.05, 0.1]:
                map_result = self.run_one_experiment(a)
                for algo_name in map_result:
                    for signal in map_result[algo_name]:
                        error = map_result[algo_name][signal] / self.steps
                        algos_map = self._update_algos_map(
                            algos_map, algo_name, signal, error)
        for algo_name in algos_map:
            Draw().plot_using_a_map(algos_map[algo_name], algo_name)
        # Store @param algos_map in a npy (binary) format.
        with open(
                os.path.dirname(os.path.abspath(__file__)) +
                "/figures/change signal/result_dict_kappa" +
                str(int(self.kappa)) + ".npy", 'wb') as f:
            np.save(f, algos_map)
        plt.xlabel("signal a")
        plt.ylabel("exact recovery")
        plt.title("Change signal with kappa " + str(int(self.kappa)))
        plt.legend()
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) +
            "/figures/change signal/comparison by change of signal kappa" +
            str(int(self.kappa)) + ".pdf")
        plt.clf()


if __name__ == "__main__":
    """To change the condition number, modify kappa.
    """
    kappa = 1.
    ChangeSignal(kappa).change_signal()
