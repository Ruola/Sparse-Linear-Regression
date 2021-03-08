import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os

from algorithms.gradient_descent import GradientDescent
from algorithms.iterative_threshold_methods import IterativeThresholdMethods
import utils.constants as constants
from utils.error import Error
from utils.generate_data import GenerateData


class Compare:
    """Do a simulation to compare gradient descent methods with IHT / HTP.
    """
    def __init__(self, design, gd_types, thres_types, other_methods):
        """Set the type of design matrix, e.g. isotropic or anisotropic.

        @param design - the type of design matrix, e.g. isotropic or anisotropic.
        @param gd_types - GD, NGD, Newton or FastNewton.
        @param thres_types - IHT or HTP.
        @param other_methods - ISTA.
        """
        self.design = design
        self.gd_types = gd_types
        self.thres_types = thres_types
        self.other_methods = other_methods
        self.steps = constants.GD_STEPS  # number of experiments
        self.num_iter = constants.GD_NUM_ITERATION  # number of iterations
        self.x_original = constants.X

    def draw_change_of_error_by_threshold(self, map_of_thres_error, type: str):
        """Get the change of testing error w.r.t. the threshold in second order methods.

        @param map_of_thres_error: keys are thresholds and values are errors.
        @param type: algorithm names, e.g. "GD+AdaIHT".
        """
        thresholds, errors = zip(*sorted(map_of_thres_error.items()))
        best_thres = min(map_of_thres_error, key=map_of_thres_error.get)
        plt.plot(thresholds,
                 errors,
                 label=type + " threshold " + str(best_thres))
        # find best threshold
        plt.scatter(best_thres,
                    map_of_thres_error[best_thres])  # draw the lowest point
        plt.xlabel("(final) threshold")
        plt.ylabel("validation error from cv")
        plt.legend()

    def compare_validation_error(self):
        """Get the change of testing error w.r.t. (final) threshold in one experiment.
        """
        y, H, self.SIGMA_half = GenerateData(self.design).generate_data()
        for gd_type in self.gd_types:
            for thres_type in self.thres_types:
                _, best_lambda, gener_error, map_of_thres_error = GradientDescent(
                ).get_errors_by_cv(self.x_original, y, H, self.num_iter,
                                   self.SIGMA_half, self.gd_type,
                                   self.thres_type, True)
                self.draw_change_of_error_by_threshold(
                    map_of_thres_error, self.gd_type + " + " + self.thres_type)
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) +
            "/figures/second order methods/error by threshold " + self.design)
        plt.clf()

    def run_one_experiment(self, dummy):
        """Run one experient working for multiprocessing.
        
        @param dummy - a meaningless variable.
        @return algo_name gener_error hashmap.
        """
        y, H, self.SIGMA_half = GenerateData(self.design).generate_data()
        algo_error_map = dict()
        for gd_type in self.gd_types:
            for thres_type in self.thres_types:
                algo_name = gd_type + "+" + thres_type
                _, best_lambda, gener_error = GradientDescent(
                ).get_errors_by_cv(self.x_original, y, H, self.num_iter,
                                   self.SIGMA_half, gd_type, thres_type, False)
                algo_error_map[algo_name] = gener_error
                print(gd_type, thres_type, best_lambda)
        for algo_name in self.other_methods:
            _, best_lambda, gener_error = IterativeThresholdMethods(
            ).get_errors_by_cv(self.x_original, y, H, self.num_iter,
                                self.SIGMA_half, algo_name, False)
            algo_error_map[algo_name] = gener_error
            print(algo_name, best_lambda)
        return algo_error_map

    def compare_gradient_descent(self):
        """Compare gradient descent, natural gd, newton with IHT / HTP.
        """
        gener_errors_matrix_map = dict()
        """
        The aim of @PARAM gener_errors_matrix_map is to store the errors of different algorithms.
        Key: "gdIHT" / "ngdIHT" / "gdHTP" etc
        Value: a matrix - each row is an experiment and each column is an iteraton
        """
        algo_gener_errors_map = dict()
        # Set initial value of gener_errors_matrix_map
        for gd_type in self.gd_types:
            for thres_type in self.thres_types:
                algo_name = gd_type + "+" + thres_type
                algo_gener_errors_map[algo_name] = np.zeros((self.num_iter))
        for algo_name in self.other_methods:
            algo_gener_errors_map[algo_name] = np.zeros((self.num_iter))

        # To multiprocess several experiments.
        pool = mp.Pool(mp.cpu_count())
        pool_result = pool.map(self.run_one_experiment, [1] * self.steps,
                               chunksize=1)
        for algo_error_map in pool_result:
            for algo_name in algo_error_map:
                algo_gener_errors_map[algo_name] += algo_error_map[algo_name]
        for algo_name in algo_gener_errors_map:  # take average
            algo_gener_errors_map[algo_name] /= self.steps

        for algo_name in algo_gener_errors_map:  # plot
            plt.plot(algo_gener_errors_map[algo_name], label=algo_name)
        plt.title("Second order methods comparison in " + self.design +
                  " design")
        plt.xlabel("#iterations")
        plt.ylabel("generalization error")
        plt.legend()
        if constants.HTP_NAME in thres_types:
            plt.savefig(
                os.path.dirname(os.path.abspath(__file__)) +
                "/figures/HTP/comparison in " + self.design + " design HTP")
        elif constants.FAST_NEWTON_NAME in self.gd_types:
            plt.savefig(
                os.path.dirname(os.path.abspath(__file__)) +
                "/figures/fast newton/comparison in " + self.design +
                " design fast newton")
        else:
            plt.savefig(
                os.path.dirname(os.path.abspath(__file__)) +
                "/figures/second order methods/comparison in " + self.design +
                " design")
        plt.clf()


if __name__ == "__main__":
    """Run the simulation.
    """
    """Change gd_types and thres_types according to the needs.    
    gd_types = (constants.GD_NAME, constants.NGD_NAME, constants.NEWTON_NAME, constants.FAST_NEWTON_NAME)
    thres_types = (constants.IHT_NAME, constants.HTP_NAME)
    other_methods = (constants.ISTA_NAME, )
    """
    gd_types = (constants.GD_NAME, constants.NEWTON_NAME, constants.FAST_NEWTON_NAME)
    thres_types = (constants.IHT_NAME, constants.HTP_NAME)
    other_methods = (constants.ISTA_NAME,)

    isotropic_obj = Compare(constants.ISOTROPIC_NAME, gd_types, thres_types,
                            other_methods)
    anisotropic_obj = Compare(constants.ANISOTROPIC_NAME, gd_types,
                              thres_types, other_methods)
    """Draw the change of testing error w.r.t. (final) threshold.
        Please comment these two lines out if you do not need it.
    """
    #isotropic_obj.compare_validation_error()
    #anisotropic_obj.compare_validation_error()
    """Draw the change of generalization error w.r.t. iterations.
    """
    isotropic_obj.compare_gradient_descent()
    anisotropic_obj.compare_gradient_descent()
