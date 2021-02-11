import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import constants
from error import Error
from generate_data import GenerateData
import numpy as np
from ista_cross_validation import Ista
from gradient_descent import GradientDescent


class IterativeThresholdMethods:
    """Do a simulation to compare the convergence speed of ISTA and IHT.
    Get the change of generalization error with respect to #iterations in ISTA and IHT.
    
    1. Find the best threshold of ISTA/IHT.
    2. Run the ISTA/IHT using the best threshold and get generalization error in each iteration.
    3. Do several experiments and take average on the error in each iteration.
    """
    def __init__(self, design="anisotropic"):
        """Set the type of design matrix and generate sparse linear regression data.
        
        @param design - "isotropic" or "anisotropic"
        """
        self.design = design
        self.N = constants.N_ITERATION  # number of iterations in ISTA or Hard Threshold
        self.n, self.p, self.s = constants.N, constants.P, constants.S
        self.x_value = constants.X_VALUE
        self.SIGMA_half = constants.SIGMA_COVAR_MATRIX_HALF[
            design]  # half of design covariance
        self.sigma = constants.SIGMA_NUMBER
        # Generate one experiment data
        self.x_original, self.y, self.H = GenerateData().generate_data(
            self.n, self.p, self.s, self.sigma, self.SIGMA_half, self.x_value)
        #self.step_size = 1/ 2. / math.ceil(max(np.linalg.eigh(np.dot(self.H.T, self.H))[0]))
        self.step_size = 1 / 2 / np.linalg.norm(
            np.dot(np.transpose(self.H), self.H), 2)

    def draw_result(self, gener_errors_ista, gener_errors_iht):
        """Draw the change of generalization error with respect to #iterations in ISTA and IHT.
        
        The figures are under ./figures/ista iht/ directory.
        """
        plt.plot(gener_errors_ista, label="ISTA")
        plt.plot(gener_errors_iht, label="IHT")
        plt.xlabel("#iterations")
        plt.ylabel("generalization error")
        plt.title("Compare convergence rate of ISTA and IHT" + self.design)
        plt.legend()
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) +
            "/figures/ista iht/convergence rate of ISTA and IHT " +
            self.design)
        plt.clf()

    def find_best_threshold_of_ISTA_IHT(self):
        """Find the best threshold of ISTA/IHT by trying thresholds and comparing results.
        
        @ return the best threshold of ISTA and the best final threshold of Ada-IHT
        """
        # ISTA
        thresholds_error_map_ista = dict(
        )  # The format is {threshold: gener error}
        for threshold in np.linspace(0, 1, 200):
            _, _, gener_errors = Ista().get_errors_by_ista(
                self.y, self.H, threshold, self.step_size, self.N,
                self.SIGMA_half, self.x_original)
            thresholds_error_map_ista[threshold] = gener_errors[-1]
        lists = sorted(thresholds_error_map_ista.items()
                       )  # sorted by key, return a list of tuples
        thresholds_ista, errors_ista = zip(*lists)
        plt.plot(thresholds_ista, errors_ista)
        # To find best lambda / threshold with lowest error
        best_error_ista, best_thres_ista = min(
            zip(errors_ista, thresholds_ista))
        plt.scatter(best_thres_ista,
                    best_error_ista)  # draw the lowest point in fig
        plt.xlabel("threshold")
        plt.ylabel("generalization error")
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) +
            "/figures/ista iht/ISTA gener error by threshold " + self.design)
        plt.clf()

        # AdaIHT
        thresholds_error_map_iht = dict()
        for threshold in np.linspace(0.005, 15, 300):
            x, final_thres, _, gener_errors = GradientDescent(
            ).solve_spare_linear_regression(self.x_original,
                                            self.y,
                                            self.H,
                                            self.N,
                                            self.SIGMA_half,
                                            threshold,
                                            gd_type="gd",
                                            iter_type="IHT")
            thresholds_error_map_iht[final_thres] = gener_errors[-1]
        lists = sorted(thresholds_error_map_iht.items()
                       )  # sorted by key, return a list of tuples
        thresholds_iht, errors_iht = zip(
            *lists)  # unpack a list of pairs into two tuples
        plt.plot(thresholds_iht, errors_iht)
        # find best threshold
        best_error_iht, best_thres_iht = min(zip(errors_iht, thresholds_iht))
        plt.scatter(best_thres_iht, best_error_iht)  # draw the lowest point
        plt.xlabel("(final) threshold")
        plt.ylabel("generalization error")
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) +
            "/figures/ista iht/AdaIHT gener error by threshold " + self.design)
        plt.clf()
        return best_thres_ista, best_thres_iht

    def compare_convergence_rate(self):
        """Get the change of generalizaton error with respect to #iteration in one experiment.
        
        @return generalization errors of each iteration in ISTA and IHT.
        """
        best_thres_ista, best_thres_iht = self.find_best_threshold_of_ISTA_IHT(
        )
        _, _, gener_errors_ista = Ista().get_errors_by_ista(
            self.y, self.H, best_thres_ista, self.step_size, self.N,
            self.SIGMA_half, self.x_original)
        _, _, _, gener_errors_iht = GradientDescent(
        ).solve_spare_linear_regression(self.x_original,
                                        self.y,
                                        self.H,
                                        self.N,
                                        self.SIGMA_half,
                                        best_thres_iht,
                                        gd_type="gd",
                                        iter_type="IHT")
        return gener_errors_ista, gener_errors_iht


if __name__ == "__main__":
    """Run several experiemnts and take the average on the errors in each iteration.
    """
    gener_errors_matrix_ista = np.zeros(
        (constants.STEPS, constants.N_ITERATION))
    gener_errors_matrix_iht = np.zeros(
        (constants.STEPS, constants.N_ITERATION))
    # Change this into "isotropic" or "anisotropic" in order to try different type of design matrix.
    design = "isotropic"
    for i in range(constants.STEPS):  # do several experiments
        gener_errors_ista, gener_errors_iht = IterativeThresholdMethods(
            design).compare_convergence_rate()
        gener_errors_matrix_ista[i] = gener_errors_ista
        gener_errors_matrix_iht[i] = gener_errors_iht
    IterativeThresholdMethods(design).draw_result(
        np.mean(gener_errors_matrix_ista, axis=0),
        np.mean(gener_errors_matrix_iht, axis=0)) # Take average
