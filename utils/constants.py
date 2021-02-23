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

STEPS = 2  # number of experiments
N_ITERATION = 200  # number of iterations in ISTA or Hard Threshold
N, P, S = 200, 1000, 10
# covariance of design matrix
temp = np.ones((P))
SIGMA_COVAR_MATRIX_HALF = {ISOTROPIC_NAME: np.diag(temp)}
temp[P // 2:] = 10
#temp[:p // 2] = 10
temp = np.random.permutation(temp)
SIGMA_COVAR_MATRIX_HALF[ANISOTROPIC_NAME] = np.diag(temp)  # half of design covariance
# noise
MU = 0
SIGMA_NUMBER = .1
# signal
X_VALUE = 1.
X = X_VALUE * np.ones((P,1))
#x[:p-s] = 0
X[S:] = 0
X = np.random.permutation(X)

ISTA_NAME = "ISTA"
IHT_NAME = "AdaIHT"
