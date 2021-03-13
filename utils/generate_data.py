import numpy as np

import utils.constants as constants


class GenerateData:
    """Generate the model.
    """
    def __init__(self, design=constants.ISOTROPIC_NAME, kappa=10., x=constants.X):
        """Initialize the model.
        
        @param design - the type of design matrix, e.g. isotropic or anisotropic.
        @param kappa - condition number of design matrix
        """
        self.design = design
        self.mu = constants.MU
        self.n, self.p, self.s = constants.N, constants.P, constants.S
        self.x = x
        # half of design covariance
        temp = np.ones((constants.P))
        if self.design == constants.ISOTROPIC_NAME:
            self.SIGMA_half = np.diag(temp)
        else:
            temp[constants.P // 2:] = kappa
            temp = np.random.permutation(temp)
            self.SIGMA_half = np.diag(temp)
        self.sigma = constants.SIGMA_NUMBER # variance of noise

    def generate_data(self):
        """Generate response and design matrix.
        
        @return response and design matrix.
        """
        H = np.dot(np.random.randn(self.n, self.p), self.SIGMA_half)
        # y = H * x + noise
        y = np.dot(H, self.x) + np.random.normal(
            self.mu, self.sigma, size=(self.n))
        return (y, H, self.SIGMA_half)
