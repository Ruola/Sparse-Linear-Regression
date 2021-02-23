import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from algorithms.iterative_threshold_methods import IterativeThresholdMethods
import utils.constants as constants
from utils.error import Error
from utils.generate_data import GenerateData


class CompareIterativeThresholdMethods:
    """Do a simulation to compare ISTA and IHT.
    """
    def __init__(self, design="anisotropic"):
        """Set the type of design matrix and generate sparse linear regression data.
        
        @param design - "isotropic" or "anisotropic"
        """
        self.design = design
        self.num_iter = constants.N_ITERATION  # number of iterations
        self.x_original = constants.X
        self.SIGMA_half = constants.SIGMA_COVAR_MATRIX_HALF[
            design]  # half of design covariance
        # Generate one experiment data
        self.y, self.H = GenerateData(design).generate_data()
        #self.step_size = 1/ 2. / math.ceil(max(np.linalg.eigh(np.dot(self.H.T, self.H))[0]))
        self.step_size = 1 / 2. / np.linalg.norm(
            np.dot(np.transpose(self.H), self.H), 2)

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


def compare_validation_error(design):
    """Get the change of testing error w.r.t. (final) threshold in one experiment.
    
    @param design: "isotropic" or "anisotropic".
    """
    obj = CompareIterativeThresholdMethods(design)
    for algo_name in (constants.IHT_NAME, constants.ISTA_NAME):
        _, best_thres_ista, gener_errors_ista, map_of_thres_error_ista = IterativeThresholdMethods(
        ).get_errors_by_cv(obj.x_original,
                           obj.y,
                           obj.H,
                           obj.num_iter,
                           obj.SIGMA_half,
                           algo_name,
                           validation_errors_needed=True)
        obj.draw_change_of_error_by_threshold(map_of_thres_error_ista,
                                              algo_name)
    plt.savefig(
        os.path.dirname(os.path.abspath(__file__)) +
        "/figures/ista iht/AdaIHT error by threshold " + obj.design)
    plt.clf()


def compare_convergence_rate(design):
    """Get the change of generalization error with respect to #iterations in ISTA and IHT.
    
    Run several experiemnts and take the average on the errors in each iteration.
    1. Find the best threshold of ISTA/AdaIHT by cross validation.
    2. Run the ISTA/IHT using the best threshold and get generalization errors in each iteration.
    3. Do several experiments and take average on the error in each iteration.
    """
    gener_errors_matrices_map = {}
    for algo_name in (constants.IHT_NAME, constants.ISTA_NAME):
        gener_errors_matrices_map[algo_name] = np.zeros(
            (constants.STEPS, constants.N_ITERATION))
    for i in range(constants.STEPS):  # do several experiments
        obj = CompareIterativeThresholdMethods(design)
        # To find the best threshold of ISTA/AdaIHT by cross validation.
        for algo_name in (constants.IHT_NAME, constants.ISTA_NAME):
            _, best_thres, gener_errors = IterativeThresholdMethods(
            ).get_errors_by_cv(obj.x_original, obj.y, obj.H, obj.num_iter,
                               obj.SIGMA_half, algo_name)
            print(algo_name, best_thres)
            # To record errors of each iteration to a row of param gener_errors_matrix.
            gener_errors_matrix = gener_errors_matrices_map[algo_name]
            gener_errors_matrix[i] = gener_errors
            gener_errors_matrices_map[algo_name] = gener_errors_matrix

    for algo_name in (constants.IHT_NAME, constants.ISTA_NAME):
        gener_errors = np.mean(gener_errors_matrices_map[algo_name], axis=0)
        plt.plot(gener_errors, label=algo_name)
    plt.xlabel("#iterations")
    plt.ylabel("generalization error")
    plt.title("Compare convergence rate of ISTA and IHT" + design)
    plt.legend()
    plt.savefig(
        os.path.dirname(os.path.abspath(__file__)) +
        "/figures/ista iht/convergence rate of ISTA and IHT " + design)
    plt.clf()


if __name__ == "__main__":
    # Change this into "isotropic" or "anisotropic" in order to try different type of design matrix.
    design = "isotropic"
    #compare_validation_error(design)
    compare_convergence_rate(design)
