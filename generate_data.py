import numpy as np

class GenerateData:

    def __init__(self):
        # mu - the expectation of gaussian
        # sigma - the std of gaussian
        self.mu = 0


    def generate_data(self, n, p, s, sigma, SIGMA_half, x_value):
        # input: 
        # n - dimension of samples
        # p - dimension of features/predictors
        # s - sparsity
        # output: x, y, H
        # y = H * x + eps
        H = np.dot(np.random.randn(n, p), SIGMA_half)
        x = x_value * np.ones((p,1))
        #x[:p-s] = 0
        x[s:] = 0
        x = np.random.permutation(x)
        y = np.dot(H, x) + np.random.normal(self.mu, sigma, size=(n, 1))
        return (x, y, H)
