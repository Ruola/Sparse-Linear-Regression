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
        return np.linalg.norm(y - np.dot(H, x), 2) ** 2 / len(y) / 2.

    def hard_threshold_algo(self, y,H,lambda_para,alpha,N):
        x = np.zeros((len(H[0]), 1))
        objective_funcs = [0] * N
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
            lambda_para = np.maximum(lambda_para/1.5, 0.5 *math.sqrt(np.log(p)/alpha))
            # print("lambda", lambda_para)
            # print("in iteration", i, "; max x:", np.max(x))
            # print("in iteration", i, "; # nonzero x:", np.count_nonzero(x))
        return (x, objective_funcs)

    def run_hard_threshold(self, x_original, y, H, N):
        #alpha = 2. * math.ceil(max(np.linalg.eigh(np.dot(H.T, H))[0]))
        alpha = 100 * 4
        # print("alpha = ", alpha)
        lambda_para = 20
        x, objective_funcs = self.hard_threshold_algo(y,H, lambda_para,alpha,N)
        # print("x max element:", np.max(x))
        #plt.plot(objective_funcs)
        #plt.plot([self.get_objective(y, H, x_original, lambda_para)] * len(objective_funcs)) # optimal
        #plt.savefig(os.path.dirname(os.path.abspath(__file__))+"/hardThreshold")
        return x, objective_funcs

if(__name__ == "__main__"):
    x_original, y, H = GenerateData().generate_data(100, 1000, 20)
    print("x original", np.max(x_original))
    x, objective_funcs = HardThreshold().run_hard_threshold(x_original, y, H)