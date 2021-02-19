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
from iht_cross_validation import AdaIht
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
        self.num_iter = constants.N_ITERATION  # number of iterations
        self.x_original = constants.X
        self.SIGMA_half = constants.SIGMA_COVAR_MATRIX_HALF[
            design]  # half of design covariance
        # Generate one experiment data
        self.y, self.H = GenerateData(design).generate_data()
        #self.step_size = 1/ 2. / math.ceil(max(np.linalg.eigh(np.dot(self.H.T, self.H))[0]))
        self.step_size = 1 / 2. / np.linalg.norm(
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
        """Find the best threshold of ISTA/IHT by cross validation.
        
        @ return the best threshold of ISTA and the best final threshold of Ada-IHT,
        and generalization errors of each iteration in ISTA and IHT.
        """
        _, best_thres_ista, _, gener_errors_ista = Ista(
        ).get_errors_by_ista_cv(self.x_original, self.y, self.H, self.num_iter,
                                self.SIGMA_half)
        _, best_thres_iht, _, gener_errors_iht = AdaIht(
        ).get_errors_by_AdaIHT_cv(self.x_original, self.y, self.H, self.num_iter,
                                  self.SIGMA_half)
        print(best_thres_ista, best_thres_iht)
        return best_thres_ista, best_thres_iht, gener_errors_ista, gener_errors_iht

if __name__ == "__main__":
    """Run several experiemnts and take the average on the errors in each iteration.
    """
    gener_errors_matrix_ista = np.zeros(
        (constants.STEPS, constants.N_ITERATION))
    gener_errors_matrix_iht = np.zeros(
        (constants.STEPS, constants.N_ITERATION))
    # Change this into "isotropic" or "anisotropic" in order to try different type of design matrix.
    design = "anisotropic"
    for i in range(constants.STEPS):  # do several experiments
        best_thres_ista, best_thres_iht, gener_errors_ista, gener_errors_iht = IterativeThresholdMethods(
            design).find_best_threshold_of_ISTA_IHT()
        gener_errors_matrix_ista[i] = gener_errors_ista
        gener_errors_matrix_iht[i] = gener_errors_iht
    IterativeThresholdMethods(design).draw_result(
        np.mean(gener_errors_matrix_ista, axis=0),
        np.mean(gener_errors_matrix_iht, axis=0))  # Take average
