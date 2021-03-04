import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    def __init__(self, design=constants.ANISOTROPIC_NAME):
        """Set the type of design matrix, e.g. isotropic or anisotropic.

        @param design - the type of design matrix, e.g. isotropic or anisotropic.
        """
        self.design = design
        self.steps = constants.GD_STEPS  # number of experiments
        self.num_iter = constants.GD_NUM_ITERATION  # number of iterations
        self.x_original = constants.X

    def draw_result(self, error_matrix, algo_name: str, error_name: str):
        """Draw the change of generalization error with respect to #iterations in ISTA and IHT.
        """
        result = np.mean(error_matrix, axis=0)
        print(algo_name + " " + error_name)
        plt.plot(result, label=algo_name + " " + error_name)
        plt.legend()
        plt.xlabel("#iterations")
        plt.ylabel("error")
        plt.title(algo_name + " " + self.design)

    def draw_change_of_error_by_threshold(self, map_of_thres_error, type: str):
        """Get the change of testing error w.r.t. the threshold in ISTA and AdaIHT.

        @param map_of_thres_error: keys are thresholds and values are errors.
        @param type: "ISTA" or "AdaIHT"
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

    def compare_validation_error(self, gd_types, thres_types):
        """Get the change of testing error w.r.t. (final) threshold in one experiment.
        
        @param design: "isotropic" or "anisotropic".
        """
        y, H, self.SIGMA_half = GenerateData(self.design).generate_data()
        for gd_type in gd_types:
            for thres_type in thres_types:
                _, best_lambda, gener_error, map_of_thres_error = GradientDescent(
                        ).get_errors_by_cv(self.x_original, y, H, self.num_iter,
                                        self.SIGMA_half, gd_type, thres_type,
                                        True)
                self.draw_change_of_error_by_threshold(map_of_thres_error,
                                                    gd_type + " + " + thres_type)
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) +
            "/figures/second order methods/error by threshold " + self.design)
        plt.clf()

    def compare_gradient_descent(self, gd_types, thres_types):
        """Compare gradient descent, natural gd, newton with IHT / HTP.
        """
        gener_errors_matrix_map = dict()
        """
        The aim of @PARAM gener_errors_matrix_map is to store the errors of different algorithms.
        Key: "gdIHT" / "ngdIHT" / "gdHTP" etc
        Value: a matrix - each row is an experiment and each column is an iteraton
        """

        # Set initial value of gener_errors_matrix_map
        for gd_type in gd_types:
            for thres_type in thres_types:
                gener_errors_matrix_map[gd_type + thres_type] = np.zeros(
                    (self.steps, self.num_iter))

        for i in range(self.steps):
            y, H = GenerateData(self.design).generate_data()
            for gd_type in gd_types:
                for thres_type in thres_types:
                    gener_errors_matrix = gener_errors_matrix_map[gd_type +
                                                                  thres_type]
                    _, best_lambda, gener_error = GradientDescent(
                    ).get_errors_by_cv(self.x_original, y, H, self.num_iter,
                                       self.SIGMA_half, gd_type, thres_type,
                                       False)
                    print(gd_type, thres_type, best_lambda)
                    gener_errors_matrix[i] = gener_error
                    gener_errors_matrix_map[gd_type +
                                            thres_type] = gener_errors_matrix
        for gd_type in gd_types:
            for thres_type in thres_types:
                self.draw_result(gener_errors_matrix_map[gd_type + thres_type],
                                 gd_type + " " + thres_type, "")
        plt.title("Second order methods comparison in " + self.design + " design")
        plt.xlabel("#iterations")
        plt.ylabel("generalization error")
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) +
            "/figures/second order methods/comparison in " + self.design + " design")
        plt.clf()


if __name__ == "__main__":
    """Run the simulation.
    """
    gd_types = (constants.GD_NAME, constants.NGD_NAME, constants.NEWTON_NAME)
    thres_types = (constants.IHT_NAME, )
    """Draw the change of testing error w.r.t. (final) threshold.
    """
    Compare(constants.ISOTROPIC_NAME).compare_validation_error(gd_types, thres_types)
    Compare(constants.ANISOTROPIC_NAME).compare_validation_error(gd_types, thres_types)
    """Draw the change of generalization error w.r.t. iterations.
    """
    Compare(constants.ISOTROPIC_NAME).compare_gradient_descent(gd_types, thres_types)
    Compare(constants.ANISOTROPIC_NAME).compare_gradient_descent(gd_types, thres_types)
