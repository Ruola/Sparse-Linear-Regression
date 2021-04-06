import numpy as np

import utils.constants as constants


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
        return np.linalg.norm(np.dot(H, x - x_hat), 2) / len(H)

    def get_gener_error(self, x, x_hat, SIGMA_half):
        """Get the generalization error.
        
        @param x - true signal
        @param x_hat - estimation of the signal
        @param SIGMA_half - half of the covariance of the design matrix
        @return ||y - Hx||_2^2 / 2 / n
        """
        return np.linalg.norm(np.dot(SIGMA_half, x - x_hat), 2)

    def get_exact_recovery(self, x, x_hat):
        """Get the exact recovery.
        
        @param x - true signal.
        @param x_hat - estimation of the signal.
        @return exact recovery
        """
        x = np.where(x > 0, 1, 0)
        x_hat = np.where(x_hat > 0, 1, 0)
        return np.linalg.norm(x - x_hat, 2)

    def get_error(self,
                  x_true,
                  x_hat,
                  error_name=constants.GENERALIZATION_ERROR_NAME,
                  SIGMA_half=None,
                  y=None,
                  H=None):
        """Get the error
        
        @param x - true signal.
        @param x_hat - estimation of the signal.
        @param error_name - prediction, generalization, and so on.
        @param SIGMA_half - half of the covariance of the design matrix.
        @param y - observation.
        @param H - design matrix.
        @return error
        """
        if error_name == constants.GENERALIZATION_ERROR_NAME:
            return self.get_gener_error(x_true, x_hat, SIGMA_half)
        elif error_name == constants.EXACT_RECOVERY_NAME:
            return self.get_exact_recovery(x_true, x_hat)
        elif error_name == constants.PREDICTION_ERROR_NAME:
            return self.get_pred_error(H, x_true, x_hat)
        else:
            return self.get_objective(y, H, x_hat)
