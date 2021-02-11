import numpy as np

class Error:
    """Contain error functions of estimation.
    """
    def get_objective(self, y, H, x):
        """It is the objective function. E.g. used in cross validation.
        
        @param y - observation / response
        @param H - design matrix
        @param x - estimation of the signal
        @return ||y - Hx||_2^2 / 2 / n
        """
        return np.linalg.norm(y - np.dot(H, x), 2)**2 / len(y) / 2.

    def get_pred_error(self, H, x, x_hat):
        """Get the prediction error.

        @param H - design matrix
        @param x - true signal
        @param x_hat - estimation of the signal
        @return ||H (x - \hat{x})||_2 / n
        """
        return np.linalg.norm(np.dot(H, x - x_hat), 2)  / len(H)

    def get_gener_error(self, x, x_hat, SIGMA_half):
        """Get the generalization error.
        
        @param x - true signal
        @param x_hat - estimation of the signal
        @param SIGMA_half - half of the covariance of the design matrix
        @return ||y - Hx||_2^2 / 2 / n
        """
        return np.linalg.norm(np.dot(SIGMA_half, x - x_hat), 2)
