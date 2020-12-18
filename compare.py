import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generate_data import GenerateData
import numpy as np
from ista import Ista
from hard_threshold import HardThreshold
from gradient_descent import GradientDescent


class Compare:
    def __init__(self, p = 200):
        self.steps = 20  # number of experiments
        self.N = 200  # number of iterations in ISTA or Hard Threshold
        self.n, self.p, self.s = 50, p, 10
        self.x_value = 1.  # theta
        temp = np.ones((self.p))
        self.design = "anisotropic"
        if not (self.design == "isotropic"):
            temp[self.p // 2:] = 10
            #temp[:self.p // 2] = 10
            #temp = np.random.permutation(temp)
        self.SIGMA = np.diag(temp)
        self.sigma = 0.1

    def draw_result(self, error_matrix, algo_name: str, error_name: str):
        result = np.mean(error_matrix, axis=0)
        print(algo_name + " " + error_name)
        #print(result)
        plt.plot(result, label=algo_name + " " + error_name)
        plt.legend()
        plt.xlabel("#iterations in algorithm")
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
                self.n, self.p, self.s, self.sigma, self.SIGMA, self.x_value)
            x_ista, ista_objectives, pred_errors, classi_errors = Ista(
            ).run_ista(x_original, y, H, self.N, self.SIGMA)
            ista_objective_matrix[i] = ista_objectives
            ista_pred_errors_matrix[i] = pred_errors
            ista_classi_errors_matrix[i] = classi_errors
            x_hard_thres, hard_thres_objectives, pred_errors_hard, classi_errors_hard = HardThreshold(
            ).run_hard_threshold(x_original, y, H, self.N, self.SIGMA)
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

    def compare_gradient_descent(self, thres_type="HTP"):
        # compare gradient descent, natural gd, newton gd algorithm
        # @PARA objective_matrix:
        # each row represents an experiment
        # and it records objection results in each iteration
        gd_classi_errors_matrix = np.zeros((self.steps, self.N))
        ngd_classi_errors_matrix = np.zeros((self.steps, self.N))
        newton_classi_errors_matrix = np.zeros((self.steps, self.N))
        for i in range(self.steps):
            x_original, y, H = GenerateData().generate_data(
                self.n, self.p, self.s, self.sigma, self.SIGMA, self.x_value)
            x_gd, gd_objectives, pred_errors, classi_errors = GradientDescent(
            ).solve_spare_linear_regression(x_original, y, H, self.N,
                                            self.SIGMA, 1.5, "gd", thres_type)
            gd_classi_errors_matrix[i] = classi_errors
            x_ngd, ngd_objectives, pred_errors_ngd, classi_errors_ngd = GradientDescent(
            ).solve_spare_linear_regression(x_original, y, H, self.N,
                                            self.SIGMA, 1.5, "ngd", thres_type)
            ngd_classi_errors_matrix[i] = classi_errors_ngd
            x_newton, newton_objectives, pred_errors_newton, classi_errors_newton = GradientDescent(
            ).solve_spare_linear_regression(x_original, y, H, self.N,
                                            self.SIGMA, 1.5, "newton",
                                            thres_type)
            newton_classi_errors_matrix[i] = classi_errors_newton
        self.draw_result(gd_classi_errors_matrix, "gd", "generalization error")
        self.draw_result(ngd_classi_errors_matrix, "ngd",
                         "generalization error")
        self.draw_result(newton_classi_errors_matrix, "newton",
                         "generalization error")
        plt.title("fd ngd newton " + self.design)
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) +
            "/compare gd ngd newton" + thres_type)
        plt.clf()

    def try_gradient_descent(self):
        # only run one experiment
        x_original, y, H = GenerateData().generate_data(
            self.n, self.p, self.s, self.sigma, self.SIGMA, self.x_value)
        x_ista, ista_objectives, pred_errors, classi_errors = Ista(
            ).run_ista(x_original, y, H, self.N, self.SIGMA)
        plt.plot(classi_errors, label="ista" + " p" + str(self.p))
        for gd_type in ("gd", "ngd", "newton"):
            for iter_type in ("IHT", "HTP"):
                if iter_type == "IHT":
                    threshold = 100
                else:
                    threshold = 100
                x, _, _, gener_errors = GradientDescent(
                ).solve_spare_linear_regression(x_original, y, H, self.N,
                                                self.SIGMA, threshold, gd_type,
                                                iter_type)
                print("p ", self.p, gd_type + "+" + iter_type + " max of x: ", np.max(x))
                plt.plot(gener_errors, label=gd_type + "+" + iter_type + " p" + str(self.p))
        plt.ylabel("generalization error")
        plt.legend()
        plt.savefig(
            os.path.dirname(os.path.abspath(__file__)) + "/gradient descent")
        #plt.clf()


if __name__ == "__main__":
    # 1. run one experiment
    #Compare().try_gradient_descent()
    # 2. different #features
    #for p in range(70, 100, 20):
    #    Compare(p).try_gradient_descent()
    # 3. run steps experiments
    Compare().compare_gradient_descent("IHT")
    Compare().compare_gradient_descent("HTP")
