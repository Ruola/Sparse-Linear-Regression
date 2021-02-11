import numpy as np
"""Store some constant variable.
"""
STEPS = 5  # number of experiments
N_ITERATION = 200  # number of iterations in ISTA or Hard Threshold
N, P, S = 50, 200, 10
X_VALUE = 1.  # true signal
temp = np.ones((P))
SIGMA_COVAR_MATRIX_HALF = {"isotropic": np.diag(temp)}
temp[P // 2:] = 10
#temp[:p // 2] = 10
temp = np.random.permutation(temp)
SIGMA_COVAR_MATRIX_HALF["anisotropic"] = np.diag(temp)  # half of design covariance
SIGMA_NUMBER = 0.1