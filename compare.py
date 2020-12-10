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
    def __init__(self):
        self.steps = 50 # number of experiments
        self.N = 300 # number of iterations in ISTA or Hard Threshold
        self.n, self.p, self.s = 50, 300, 10
        self.x_value = 1. # theta
        temp = np.ones((self.p))
        self.design = "anisotropic"
        if not (self.design ==  "isotropic"):
            temp[:self.p//2] = 10
            #temp = np.random.permutation(temp)
        self.SIGMA = np.diag(temp)
        self.sigma = 0.1

    def draw_result(self, error_matrix, algo_name: str, error_name: str):
        result = np.mean(error_matrix, axis=0)
        print(algo_name + " " + error_name)
        print(result)
        plt.plot(result, label = algo_name + " " + error_name)
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
            x_original, y, H = GenerateData().generate_data(self.n, self.p, self.s, self.sigma, self.SIGMA, self.x_value)
            x_ista, ista_objectives, pred_errors,classi_errors = Ista().run_ista(x_original, y, H, self.N, self.SIGMA)
            ista_objective_matrix[i] = ista_objectives
            ista_pred_errors_matrix[i] = pred_errors
            ista_classi_errors_matrix[i] = classi_errors
            x_hard_thres, hard_thres_objectives, pred_errors_hard,classi_errors_hard  = HardThreshold().run_hard_threshold(x_original, y, H, self.N, self.SIGMA)
            hard_objective_matrix[i] = hard_thres_objectives
            hard_thres_pred_errors_matrix[i] = pred_errors_hard
            hard_thres_classi_errors_matrix[i] = classi_errors_hard
        self.draw_result(ista_pred_errors_matrix, "ISTA", "prediction error")
        self.draw_result(ista_classi_errors_matrix, "ISTA", "classification error")
        self.draw_result(hard_thres_pred_errors_matrix, "hard thres", "prediction error")
        self.draw_result(hard_thres_classi_errors_matrix, "hard thres", "classification error")
        plt.savefig(os.path.dirname(os.path.abspath(__file__))+"/compare soft and hard")
        plt.clf()

    def compare_gradient_descent(self):
        # compare gradient descent, natural gd, newton gd algorithm
        # @PARA objective_matrix:
        # each row represents an experiment
        # and it records objection results in each iteration
        gd_objective_matrix = np.zeros((self.steps, self.N))
        gd_pred_errors_matrix = np.zeros((self.steps, self.N))
        gd_classi_errors_matrix = np.zeros((self.steps, self.N))
        ngd_objective_matrix = np.zeros((self.steps, self.N))
        ngd_pred_errors_matrix = np.zeros((self.steps, self.N))
        ngd_classi_errors_matrix = np.zeros((self.steps, self.N))
        newton_objective_matrix = np.zeros((self.steps, self.N))
        newton_pred_errors_matrix = np.zeros((self.steps, self.N))
        newton_classi_errors_matrix = np.zeros((self.steps, self.N))
        for i in range(self.steps):
            x_original, y, H = GenerateData().generate_data(self.n, self.p, self.s, self.sigma, self.SIGMA, self.x_value)
            x_gd, gd_objectives, pred_errors,classi_errors = GradientDescent().run_gradient_descent(x_original, y, H, self.N, self.SIGMA, "gd")
            gd_objective_matrix[i] = gd_objectives
            gd_pred_errors_matrix[i] = pred_errors
            gd_classi_errors_matrix[i] = classi_errors
            x_ngd, ngd_objectives, pred_errors_ngd,classi_errors_ngd  = GradientDescent().run_gradient_descent(x_original, y, H, self.N, self.SIGMA, "ngd")
            ngd_objective_matrix[i] = ngd_objectives
            ngd_pred_errors_matrix[i] = pred_errors_ngd
            ngd_classi_errors_matrix[i] = classi_errors_ngd
            x_newton, newton_objectives, pred_errors_newton,classi_errors_newton  = GradientDescent().run_gradient_descent(x_original, y, H, self.N, self.SIGMA, "newton")
            newton_objective_matrix[i] = newton_objectives
            newton_pred_errors_matrix[i] = pred_errors_newton
            ngd_classi_errors_matrix[i] = classi_errors_ngd
        self.draw_result(gd_pred_errors_matrix, "gd", "prediction error")
        self.draw_result(gd_classi_errors_matrix, "gd", "classification error")
        self.draw_result(ngd_pred_errors_matrix, "ngd", "prediction error")
        self.draw_result(ngd_classi_errors_matrix, "ngd", "classification error")
        self.draw_result(newton_pred_errors_matrix, "newton", "prediction error")
        self.draw_result(newton_classi_errors_matrix, "newton", "classification error")
        plt.title("fd ngd newton " + self.design)
        plt.savefig(os.path.dirname(os.path.abspath(__file__))+"/compare gd ngd newton")
        plt.clf()

    def try_gradient_descent(self):
        # to debug
        # only run one experiment
        x_original, y, H = GenerateData().generate_data(self.n, self.p, self.s, self.sigma, self.SIGMA, self.x_value)
        x_gd, objective_funcs_gd, pred_errors_gd, gener_errors_gd = GradientDescent().run_gradient_descent(x_original, y, H, self.N, self.SIGMA, "gd")
        x_ngd, objective_funcs_ngd, pred_errors_ngd, gener_errors_ngd = GradientDescent().run_gradient_descent(x_original, y, H, self.N, self.SIGMA, "ngd")
        x_newton, objective_funcs_newton, pred_errors_newton, gener_errors_newton = GradientDescent().run_gradient_descent(x_original, y, H, self.N, self.SIGMA, "newton")
        print("max of x: ", np.max(x_gd))
        print("max of x: ", np.max(x_ngd))
        print("max of x: ", np.max(x_newton))
        print("objective_funcs_gd", objective_funcs_gd)
        print()
        print("objective_funcs_ngd", objective_funcs_ngd)
        print()
        print("objective_funcs_newton", objective_funcs_newton)
        #plt.plot(objective_funcs_gd, label="gd objective")
        #plt.plot(pred_errors_gd, label="gd prediction error")
        plt.plot(gener_errors_gd, label="gd generalization error")
        plt.legend()
        #plt.savefig(os.path.dirname(os.path.abspath(__file__))+"/gd")
        #plt.clf()
        #plt.plot(objective_funcs_ngd, label="ngd objective")
        #plt.plot(pred_errors_ngd, label="ngd prediction error")
        plt.plot(gener_errors_ngd, label="ngd generalization error")
        plt.legend()
        #plt.savefig(os.path.dirname(os.path.abspath(__file__))+"/ngd")
        #plt.clf()
        #plt.plot(objective_funcs_newton, label="newton objective")
        #plt.plot(pred_errors_newton, label="newton prediction error")
        plt.plot(gener_errors_newton, label="newton generalization error")
        plt.legend()
        plt.savefig(os.path.dirname(os.path.abspath(__file__))+"/newton")
        plt.clf()
if __name__ == "__main__":
    Compare().try_gradient_descent()
