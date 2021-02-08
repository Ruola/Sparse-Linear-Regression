import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generate_data import GenerateData


class HardThreshold:
    def get_objective(self, y, H, x, lambda_para):
        # 1/n/2*(norm2(y-H*x)^2 + lambda*norm0(x))
        return np.linalg.norm(y - np.dot(H, x), 2)**2 / len(y) / 2.

    def get_pred_error(self, H, x, x_hat):
        return np.linalg.norm(np.dot(H, x - x_hat), 2) / len(H)

    def get_classi_error(self, x, x_hat, SIGMA_half):
        return np.linalg.norm(np.dot((SIGMA_half), x - x_hat), 2)

    def hard_threshold_algo(self, y, H, lambda_para, alpha, N, SIGMA_half, x_true):
        x = np.zeros((len(H[0]), 1))
        objective_funcs = [0] * N
        pred_errors = [0] * N
        classi_errors = [0] * N
        p = len(H[0])
        n = len(H)
        # print("n, p", n, p)
        for i in range(N):
            # stopping criteria
            #if np.linalg.norm(y - np.dot(H, x), 2) < 1e-5 * np.count_nonzero(x):
            #   break
            # hard(x + 1/alpha/n H.T(y-Hx), lambda)
            x = x + np.dot(np.transpose(H), y - np.dot(H, x)) / alpha
            x[np.abs(x) <= lambda_para] = 0
            objective_funcs[i] = self.get_objective(y, H, x, lambda_para)
            pred_errors[i] = self.get_pred_error(H, x_true, x)
            classi_errors[i] = self.get_classi_error(x_true, x, SIGMA_half)
            lambda_para = np.maximum(lambda_para / 1.5,
                                     0.5 * math.sqrt(np.log(p) / alpha))
            # print("lambda", lambda_para)
            # print("in iteration", i, "; max x:", np.max(x))
            # print("in iteration", i, "; # nonzero x:", np.count_nonzero(x))
        return (x, objective_funcs, pred_errors, classi_errors)

    def run_hard_threshold(self, x_original, y, H, N, SIGMA_half):
        #alpha = 5. * math.ceil(max(np.linalg.eigh(np.dot(H.T, H))[0]))
        alpha = 1000
        # print("alpha = ", alpha)
        lambda_para = 1
        x, objective_funcs, pred_errors, classi_errors = self.hard_threshold_algo(
            y, H, lambda_para, alpha, N, SIGMA_half, x_original)
        # print("x max element:", np.max(x))
        #plt.plot(objective_funcs)
        #plt.plot([self.get_objective(y, H, x_original, lambda_para)] * len(objective_funcs)) # optimal
        #plt.savefig(os.path.dirname(os.path.abspath(__file__))+"/hardThreshold")
        return x, objective_funcs, pred_errors, classi_errors
