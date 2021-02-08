import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generate_data import GenerateData
import numpy as np
from ista_cross_validation import Ista
from adaptive_iterative_hard_threshold import HardThreshold
from gradient_descent import GradientDescent


class Compare:
    def __init__(self, design="anisotropic"):
        self.steps = 20  # number of experiments
        self.N = 200  # number of iterations in ISTA or Hard Threshold
        self.n, self.p, self.s = 50, 200, 10
        self.x_value = 1.  # theta
        temp = np.ones((self.p))
        self.design = design  # default value is "anisotropic"
        if not (self.design == "isotropic"):
            temp[self.p // 2:] = 10
            #temp[:self.p // 2] = 10
            temp = np.random.permutation(temp)
        self.SIGMA_half = np.diag(temp)  # half of design covariance
        self.sigma = 0.1

    def draw_result(self, error_matrix, algo_name: str, error_name: str):
        result = np.mean(error_matrix, axis=0)
        print(algo_name + " " + error_name)
        plt.plot(result, label=algo_name + " " + error_name)
        plt.legend()
        plt.xlabel("#iterations")
        plt.ylabel("error")
        plt.title(algo_name + " " + self.design)

    def compare_soft_and_hard(self):
        # compare iterative soft/hard threshold algorithm
        # @PARA hard_objective_matrix:
        # each row represents an experiment
        # and it records objection results in each iteration
        ista_objective_matrix = np.zeros((self.steps, self.N))
        ista_pred_errors_matrix = np.zeros((self.steps, self.N))
        ista_classi_errors_matrix = np.zeros((self.steps, self.N))
        hard_objective_matrix = np.zeros((self.steps, self.N))
        hard_thres_pred_errors_matrix = np.zeros((self.steps, self.N))
        hard_thres_classi_errors_matrix = np.zeros((self.steps, self.N))
        for i in range(self.steps):
            x_original, y, H = GenerateData().generate_data(
                self.n, self.p, self.s, self.sigma, self.SIGMA_half,
                self.x_value)
            x_ista, ista_objectives, pred_errors, classi_errors = Ista(
            ).run_ista(x_original, y, H, self.N, self.SIGMA_half)
            ista_objective_matrix[i] = ista_objectives
            ista_pred_errors_matrix[i] = pred_errors
            ista_classi_errors_matrix[i] = classi_errors
            x_hard_thres, hard_thres_objectives, pred_errors_hard, classi_errors_hard = HardThreshold(
            ).run_hard_threshold(x_original, y, H, self.N, self.SIGMA_half)
            hard_objective_matrix[i] = hard_thres_objectives
            hard_thres_pred_errors_matrix[i] = pred_errors_hard
            hard_thres_classi_errors_matrix[i] = classi_errors_hard
        self.draw_result(ista_pred_errors_matrix, "ISTA", "prediction error")
        self.draw_result(ista_classi_errors_matrix, "ISTA",
                         "classification error")
        self.draw_result(hard_thres_pred_errors_matrix, "hard thres",
                         "prediction error")
        self.draw_result(hard_thres_classi_errors_matrix, "hard thres",
                         "classification error")
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) +
            "/compare soft and hard")
        plt.clf()

    def compare_gradient_descent(self):
        # compare gradient descent, natural gd, newton gd algorithm
        # @PARA objective_matrix:
        # each row represents an experiment
        # and it records objection results in each iteration
        gd_types = ("gd", "ngd", "newton")
        thres_types = ("ISTA", "IHT")
        gener_errors_matrix_map = dict()
        ista_cv_name = "ISTA cv"
        gener_errors_matrix_map[ista_cv_name] = np.zeros((self.steps, self.N))
        for gd_type in gd_types:
            for thres_type in thres_types:
                gener_errors_matrix_map[gd_type+thres_type] = np.zeros((self.steps, self.N))
        for i in range(self.steps):
            x_original, y, H = GenerateData().generate_data(
                self.n, self.p, self.s, self.sigma, self.SIGMA_half,
                self.x_value)
            _, _, _, _, gener_error_ista = Ista(
                ).run_ista(x_original, y, H, self.N, self.SIGMA_half)
            gener_errors_matrix_ista = gener_errors_matrix_map[ista_cv_name]
            gener_errors_matrix_ista[i] = gener_error_ista
            for gd_type in gd_types:
                for thres_type in thres_types:
                    gener_errors_matrix = gener_errors_matrix_map[gd_type+thres_type]
                    _, _, _, _, gener_error = GradientDescent(
                    ).solve_spare_linear_regression(x_original, y, H, self.N,
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
            "/comparison in " + self.design + " design")
        plt.clf()

    def compare_convergence_rate_ISTA_IHT(self):
        # To compare Ada-ISTA and Ada-IHT in terms of convergence rate and (an)isotropic design.
        ista_classi_errors_matrix = np.zeros((self.steps, self.N))
        # ista_classi_errors_matrix / iht_classi_errors_matrix: each row represents an experiment
        # and it records objection results in each iteration.
        iht_classi_errors_matrix = np.zeros((self.steps, self.N))
        newton_classi_errors_matrix = np.zeros((self.steps, self.N))
        for i in range(self.steps):
            x_original, y, H = GenerateData().generate_data(
                self.n, self.p, self.s, self.sigma, self.SIGMA_half,
                self.x_value)
            x_ista, _, _, _, classi_errors = GradientDescent(
            ).solve_spare_linear_regression(x_original, y, H, self.N,
                                            self.SIGMA_half, 1.5, "gd", "ISTA")
            ista_classi_errors_matrix[i] = classi_errors
            x_iht, _, _, _, classi_errors_iht = GradientDescent(
            ).solve_spare_linear_regression(x_original, y, H, self.N,
                                            self.SIGMA_half, 1.5, "gd", "IHT")
            iht_classi_errors_matrix[i] = classi_errors_iht
        self.draw_result(ista_classi_errors_matrix, "ista",
                         "generalization error")
        self.draw_result(iht_classi_errors_matrix, "iht",
                         "generalization error")
        plt.title("ISTA IHT + adaptive + " + self.design)
        plt.xlabel("#iterations")
        plt.ylabel("generalization error")
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) +
            "/compare ISTA IHT convergence rate" + self.design)
        plt.clf()

    def change_threshold_of_ISTA_IHT(self):
        # To compare ISTA and AIHT by the change of threshold
        # 1. generate data
        x_original, y, H = GenerateData().generate_data(
            self.n, self.p, self.s, self.sigma, self.SIGMA_half, self.x_value)
        # 2. ISTA
        x_ista, thres_ista, ista_objectives, pred_errors, gener_errors = Ista(
        ).run_ista(x_original, y, H, self.N, self.SIGMA_half)
        plt.clf()
        plt.scatter(thres_ista,
                    gener_errors[-1],
                    label="ista + cross validation")
        # 3. AIHT
        gd_type = "gd"
        iter_type = "IHT"
        for threshold in np.linspace(0.1, 10, 100):
            x, thres_AIHT, _, _, gener_errors = GradientDescent(
            ).solve_spare_linear_regression(x_original, y, H, self.N,
                                            self.SIGMA_half, threshold,
                                            gd_type, iter_type)
            plt.scatter(thres_AIHT, gener_errors[-1])
        plt.xlabel("(final) threshold")
        plt.ylabel("generalization error")
        plt.legend()
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) +
            "/compare ISTA and AIHT by the change of threshold")
        plt.clf()


if __name__ == "__main__":
    # To run steps experiments
    #Compare().change_threshold_of_ISTA_IHT()
    #Compare().change_threshold_of_ISTA_IHT()
    #Compare("isotropic").compare_convergence_rate_ISTA_IHT()
    #Compare("anisotropic").compare_convergence_rate_ISTA_IHT()
    Compare("isotropic").compare_gradient_descent()
    Compare("anisotropic").compare_gradient_descent()
