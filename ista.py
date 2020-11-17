import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cross_validation import CrossValidation as cv
from generate_data import GenerateData

class Ista:
    def get_objective(self, y, H, x, lambda_para):
        # 1/n/2*(norm2(y-H*x)^2 + lambda*norm1(x))
        return np.linalg.norm(y - np.dot(H, x), 2) ** 2/len(y) / 2.

    def soft_threshold(self, y,H,lambda_para,alpha,N):
        x = np.zeros((len(H[0]), 1))
        for _ in range(N):
            # soft(x + 1/alpha/n H.T(y-Hx), lambda)
            # soft(temp, lambda)
            temp = x + np.dot(np.transpose(H), y - np.dot(H, x)) / alpha / len(y)
            x = temp - lambda_para*np.sign(temp)
            x[x*np.sign(temp) < 0] = 0
        return x

    def ista(self, y,H,lambda_para,alpha,N):
        # input:
        # y - observed signal
        # H - design matrix
        # lambda_para - regularization para
        # alpha >= max eigen (H'H)
        # N - max number of iterations
        # output:
        # x - recovered signal,
        x = np.zeros((len(H[0]), 1))
        objective_funcs = [0] * N
        temp = np.zeros(x.size)
        for i in range(N):
            # soft(x + 1/alpha/n H.T(y-Hx), lambda)
            # soft(temp, lambda)
            temp = x + np.dot(np.transpose(H), y - np.dot(H, x)) / alpha / len(y)
            x = temp - lambda_para*np.sign(temp)
            x[x*np.sign(temp) < 0] = 0
            # x[abs(temp) < lambda] = 0
            objective_funcs[i] = self.get_objective(y, H, x, lambda_para)
        return (x, objective_funcs)

    def run_ista(self, x_original, y, H, N):
        alpha = 2. * math.ceil(max(np.linalg.eigh(np.dot(H.T, H))[0])) / len(y)
        # print("alpha = ", alpha)
        # tuning lambda
        objective_f = self.get_objective
        algo_f = lambda _y, _H, _para: self.soft_threshold(_y,_H,_para,alpha,N)
        cv_obj = cv(objective_f, algo_f, y, H, lambda_min=0.01, lambda_max=1, params_count=10, k=5)
        best_lambda = cv_obj.tune_para()
        # print("best lambda", best_lambda)
        x, best_objective_funcs = self.ista(y,H,best_lambda,alpha,N)
        # print("x max element:", np.max(x))
        #plt.plot(best_objective_funcs)
        #plt.plot([self.get_objective(y, H, x_original, best_lambda)] * len(best_objective_funcs)) # optimal
        #plt.savefig(os.path.dirname(os.path.abspath(__file__))+"/ista")
        return x, best_objective_funcs

if(__name__ == "__main__"):
    x_original, y, H = GenerateData().generate_data(100, 1000, 20) # n = 100, p = 1000
    print("x original", np.max(x_original))
    x, objective_funcs = Ista().run_ista(x_original, y, H)
