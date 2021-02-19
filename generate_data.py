import numpy as np

import constants


class GenerateData:
    """Generate the model.
    """
    def __init__(self, design="isotropic"):
        """Initialize the model.
        
        @param design - the type of design matrix, e.g. isotropic or anisotropic.
        """
        self.design = design
        self.mu = constants.MU
        self.steps = constants.STEPS  # number of experiments
        self.N = constants.N_ITERATION  # number of iterations in ISTA or Hard Threshold
        self.n, self.p, self.s = constants.N, constants.P, constants.S
        self.x = constants.X
        self.SIGMA_half = constants.SIGMA_COVAR_MATRIX_HALF[
            design]  # half of design covariance
        self.sigma = constants.SIGMA_NUMBER

    def generate_data(self):
        """Generate response and design matrix.
        
        @return response and design matrix.
        """
        H = np.dot(np.random.randn(self.n, self.p), self.SIGMA_half)
        # y = H * x + noise
        y = np.dot(H, self.x) + np.random.normal(
            self.mu, self.sigma, size=(self.n, 1))
        return (y, H)
