import numpy as np

class GenerateData:

    def __init__(self):
        # mu - the expectation of gaussian
        # sigma - the std of gaussian
        self.mu = 0
        self.sigma = 0.1 # default

    def generate_data(self, n, p, s, sigma):
        # input: 
        # n - dimension of samples
        # p - dimension of features/predictors
        # s - sparsity
        # output: x, y, H
        # y = H * x + eps
        self.sigma = sigma
        H = np.random.randn(n, p)
        x = 1. * np.ones((p,1))
        x[s:] = 0
        x = np.random.permutation(x)
        y = np.dot(H, x) + np.random.normal(self.mu, self.sigma, size=(n, 1))
        return (x, y, H)
