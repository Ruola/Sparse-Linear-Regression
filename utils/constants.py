import numpy as np
"""Store some constant variable.

@param STEPS - number of experiments
@param N_ITERATION - max number of iterations
@param N - number of responses for obeservations
@param P - number of features
@param S - sparsity, num of nonzero elements of signal
@param SIGMA_COVAR_MATRIX_HALF - half of covariance of design matrix
@param MU - expectation of noise.
@param SIGMA_NUMBER - variance of noise
@param X_VALUE - value of elements of true signal
@param X - true signal
"""
ISOTROPIC_NAME = "isotropic"
ANISOTROPIC_NAME = "anisotropic"

STEPS = 20  # number of experiments
N_ITERATION = 400  # number of iterations in ISTA or Hard Threshold
GD_STEPS = 10  # number of experiments in compare_gradient_descent
GD_NUM_ITERATION = 400  # number of iterations in ISTA or Hard Threshold
FAST_NEWTON_NUM_GD = 10 # number of gradient descent steps in fast newton
N, P, S = 200, 1000, 10
# noise
MU = 0
SIGMA_NUMBER = .1
# signal
X_VALUE = 1.
X = X_VALUE * np.ones((P))
#x[:p-s] = 0
X[S:] = 0
X = np.random.permutation(X)

FAST_NEWTON_NAME = "FastNewton"
ISTA_NAME = "ISTA"
IHT_NAME = "AdaIHT"
HTP_NAME = "AdaHTP"

GD_NAME = "gd"
NGD_NAME = "ngd"
NEWTON_NAME = "newton"
