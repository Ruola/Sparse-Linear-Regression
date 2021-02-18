import numpy as np
"""Store some constant variable.
"""
STEPS = 20  # number of experiments
N_ITERATION = 200  # number of iterations in ISTA or Hard Threshold
N, P, S = 200, 1000, 10

temp = np.ones((P))
SIGMA_COVAR_MATRIX_HALF = {"isotropic": np.diag(temp)}
temp[P // 2:] = 10
#temp[:p // 2] = 10
temp = np.random.permutation(temp)
SIGMA_COVAR_MATRIX_HALF["anisotropic"] = np.diag(temp)  # half of design covariance
SIGMA_NUMBER = .1
X_VALUE = 1.  # true signal
X = X_VALUE * np.ones((P,1))
#x[:p-s] = 0
X[S:] = 0
X = np.random.permutation(X)