import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from error import Error
from generate_data import GenerateData
import constants
from ista_cross_validation import Ista
from gradient_descent import GradientDescent


class Compare:
    """Do a simulation to compare gradient descent methods with IHT / HTP.
    """
    def __init__(self, design="anisotropic"):
        """Set the type of design matrix, e.g. isotropic or anisotropic.

        @param design - the type of design matrix, e.g. isotropic or anisotropic.
        """
        self.design = design
        self.steps = constants.STEPS  # number of experiments
        self.num_iter = constants.N_ITERATION  # number of iterations
        self.x_original = constants.X
        self.SIGMA_half = constants.SIGMA_COVAR_MATRIX_HALF[design]  # half of design covariance

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

    def compare_gradient_descent(self):
        """Compare gradient descent, natural gd, newton with IHT / HTP.
        """
        gd_types = ("gd", "ngd", "newton")
        thres_types = ("IHT", "HTP")
        gener_errors_matrix_map = dict()
        """
        The aim of @PARAM gener_errors_matrix_map is to store the errors of different algorithms.
        Key: "gdIHT" / "ngdIHT" / "gdHTP" etc
        Value: a matrix - each row is an experiment and each column is an iteraton
        """
        ista_cv_name = "ISTA cv" # a key of gener_errors_matrix_map
        gener_errors_matrix_map[ista_cv_name] = np.zeros((self.steps, self.num_iter))

        # Set initial value of gener_errors_matrix_map
        for gd_type in gd_types:
            for thres_type in thres_types:
                gener_errors_matrix_map[gd_type+thres_type] = np.zeros((self.steps, self.num_iter))

        for i in range(self.steps):
            y, H = GenerateData(self.design).generate_data()
            # ISTA and lambda/threshold is selected by cross validation
            _, _, _, gener_error_ista = Ista(
                ).get_errors_by_ista_cv(self.x_original, y, H, self.num_iter, self.SIGMA_half)
            gener_errors_matrix_ista = gener_errors_matrix_map[ista_cv_name]
            gener_errors_matrix_ista[i] = gener_error_ista
            for gd_type in gd_types:
                for thres_type in thres_types:
                    gener_errors_matrix = gener_errors_matrix_map[gd_type+thres_type]
                    _, _, _, gener_error = GradientDescent(
                    ).solve_spare_linear_regression(self.x_original, y, H, self.num_iter,
                                                    self.SIGMA_half, 1.5, gd_type,
                                                    thres_type)
                    gener_errors_matrix[i] = gener_error
                    gener_errors_matrix_map[gd_type+thres_type] = gener_errors_matrix
        for gd_type in gd_types:
            for thres_type in thres_types:
                self.draw_result(gener_errors_matrix_map[gd_type+thres_type], gd_type+" "+thres_type, "")
        self.draw_result(gener_errors_matrix_map[ista_cv_name], "ISTA cross validation", "")
        plt.title("comparison in " + self.design + " design")
        plt.xlabel("#iterations")
        plt.ylabel("generalization error")
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) +
            "/figures/comparison in " + self.design + " design")
        plt.clf()

if __name__ == "__main__":
    """Run the simulation.
    """
    Compare("isotropic").compare_gradient_descent()
    #Compare("anisotropic").compare_gradient_descent()
