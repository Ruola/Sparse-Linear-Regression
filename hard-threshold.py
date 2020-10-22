import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_data(n, p, s):
    # input: 
    # n - dimension of samples
    # p - dimension of features/predictors
    # mu - the expectation of gaussian
    # sigma - the std of gaussian
    # output: x, y, H
    # y = H * x + eps
    H = np.random.randn(n, p)
    #x = np.random.rand(p, 1)
    x = 1. * np.ones((p,1))
    x[s:] = 0
    x = np.random.permutation(x)
    y = np.dot(H, x) + np.random.normal(0, 0.1, size=(n, 1))
    return (x, y, H)

def get_objective(y, H, x, lambda_para):
    # 1/n/2*(norm2(y-H*x)^2 + lambda*norm0(x))
    return np.linalg.norm(y - np.dot(H, x), 2) ** 2 / len(y) / 2.

def hard_threshold_algo(y,H,lambda_para,alpha,N):
    x = np.zeros((len(H[0]), 1))
    objective_funcs = [0] * N
    for i in range(N):
        # stopping criteria
        if np.linalg.norm(y - np.dot(H, x), 2) < 1e-5 * np.count_nonzero(x):
            break
        # hard(x + 1/alpha/n H.T(y-Hx), lambda)
        x = x + np.dot(np.transpose(H), y - np.dot(H, x)) / alpha / len(y)
        x[np.abs(x) <= lambda_para] = 0
        objective_funcs[i] = get_objective(y, H, x, lambda_para)
        lambda_para /= 2
        print("in iteration", i, "; max x:", np.max(x))
    return (x, objective_funcs)

if(__name__ == "__main__"):
    x_original, y, H = generate_data(100, 1000, 20)
    print("x original", np.max(x_original))
    alpha = 2. * math.ceil(max(np.linalg.eigh(np.dot(H.T, H))[0])) / len(y)
    print("alpha = ", alpha)
    N = 200
    # tuning lambda
    lambda_para = 20
    x, objective_funcs = hard_threshold_algo(y,H, lambda_para,alpha,N)
    print("x max element:", np.max(x))
    plt.plot(objective_funcs)
    plt.plot([get_objective(y, H, x_original, lambda_para)] * len(objective_funcs)) # optimal
    plt.savefig(os.path.dirname(os.path.abspath(__file__))+"/hardThreshold")

