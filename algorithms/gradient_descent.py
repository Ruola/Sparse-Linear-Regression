import numpy as np
from scipy.linalg import pinvh
import utils.constants as constants
from utils.cross_validation import CrossValidation as cv
from utils.error import Error


class GradientDescent:
    """Use gradient descent (gd), natural gd, newton or fast newton method with IHT/HTP to solve sparse linear regression.
    """
    def __init__(self, fast_newton_num_gd=constants.FAST_NEWTON_NUM_GD):
        """@param fast_newton_num_gd - the number of gradient descent iterations before thresholding in Fast Newton.
        """
        self.fast_newton_num_gd = fast_newton_num_gd

    def get_gd_step_size(self, H, gd_type: str, inv_sigma):
        """Compute the step size of gradient descent.
        
        @param H - the design matrix.
        @param gd_type - includes gradient descent (gd), natural gd (ngd),
                newton method (newton).
        @param inv_sigma - inverse of the design covariance matrix.
        @return
        """
        if gd_type == constants.GD_NAME or gd_type == constants.FAST_NEWTON_NAME:  # gradient descent
            step_size = 1 / 2 / np.linalg.norm(np.dot(np.transpose(H), H), 2)
        elif gd_type == constants.NGD_NAME:  # natural gradient descent
            step_size = 1 / 2 / np.linalg.norm(
                np.dot(inv_sigma, np.dot(np.transpose(H), H)), 2)
        else:  # newton
            step_size = 1 / 2
        return step_size

    def get_gradient_descent_step(self, x, y, H, gd_type: str, inv_sigma,
                                  step_size):
        """Compute the step size of gradient descent.
        
        @param x - signal estimation.
        @param y - the observations/response.
        @param H - the design matrix.
        @param gd_type - includes gradient descent (gd), natural gd (ngd),
                newton method (newton).
        @param inv_sigma - inverse of the design covariance matrix.
        @param step_size - gradient descent step size.
        @return the gradient descent step.
        """
        if gd_type == constants.GD_NAME or gd_type == constants.FAST_NEWTON_NAME:
            return step_size * np.dot(np.transpose(H), y - np.dot(H, x))
        elif gd_type == constants.NGD_NAME:
            return step_size * np.dot(np.dot(inv_sigma, np.transpose(H)),
                                      y - np.dot(H, x))
        else:
            return step_size * np.dot(
                np.dot(pinvh(np.dot(np.transpose(H), H)), np.transpose(H)),
                y - np.dot(H, x))

    def _insert_zero(self, x, indices_removed):
        """A function to add zero elements into x as the list of indices removed previously.
        
        @param x - signal estimation.
        @param indices_removed - the list of indices removed previously.
        @return the updated signal estimation.
        """
        for i in indices_removed:
            x = np.insert(x, i, 0)
        return x

    def update_signal_estimation(self, x, y, H, lambda_step, gd_type,
                                 iter_index, iter_type):
        """Get the estimation by IHT or HTP.
        This step follows the gradient descent step. 
        Update the estimation by pluging coordinates that are smaller than the threshold to zero.
        
        @param x - signal estimation.
        @param y - the observations/response.
        @param H - the design matrix.
        @param lambda_step - the threshold.
        @param gd_type - includes gradient descent, natural gd, newton method, fast newton.
        @param iter_type - includes iterative hard threshold (IHT), hard threshold pursuit (HTP).
        """
        if gd_type == constants.FAST_NEWTON_NAME:
            if iter_index % self.fast_newton_num_gd == 0:
                x[np.abs(x) < lambda_step] = 0
            return x
        if iter_type == constants.IHT_NAME:  # iterative hard threshold
            x[np.abs(x) < lambda_step] = 0
        else:  # HTP: hard threshold pursuit
            """x = (H_sparse^T * H_sparse)^(-1) * H_sparse^T * y
            """
            indices_removed = np.argwhere(np.abs(x) < lambda_step)
            if len(indices_removed) != len(x):
                # in case H'H is 0 * 0 dimension
                H_sparse = np.delete(np.copy(H), indices_removed, 1)
                x_temp = np.dot(
                    pinvh(np.dot(np.transpose(H_sparse), H_sparse)),
                    np.dot(np.transpose(H_sparse), y))
                x_temp = x_temp.reshape(-1)
                x = self._insert_zero(x_temp, indices_removed)
        return x

    def get_estimation(self,
                       x_original,
                       y,
                       H,
                       num_iter,
                       SIGMA_half,
                       gd_type: str,
                       iter_type: str,
                       final_threshold: float,
                       errors_needed=False):
        """Use gradient descent (gd), natural gd or newton method with IHT/HTP to solve sparse linear regression.
        
        @param x_original - in order to compute error,  we need the original/true signal.
        @param y - the observations/response
        @param H - the design matrix
        @param num_iter - the number of iterations
        @param SIGMA_half - half of the covariance of design matrix
        @param gd_type - includes gradient descent, natural gd, newton method, fast newton.
        @param iter_type - includes iterative hard threshold (IHT), hard threshold pursuit (HTP)
        @param final_threshold - the lowest bound of threshold.
        @param errors_needed - If True, return generalization errors of each iteration.
        @return x - estimation \hat{x};
                gener_errors - generalization errors of estimation in each iteration
        """
        x = np.zeros((constants.P))  # initial value of estimation
        gener_errors = [
            0
        ] * num_iter  # record generalization errors of estimation in each iteration
        inv_sigma = np.diag(1. / (np.diag(SIGMA_half)**2))
        # Define step size of gradient descent step.
        step_size = self.get_gd_step_size(H, gd_type, inv_sigma)
        lambda_step = np.max(
            self.get_gradient_descent_step(x, y, H, gd_type, inv_sigma,
                                           step_size))  #threshold

        for i in range(num_iter):
            # Gradient descent step: gd, ngd, newton.
            x = x + self.get_gradient_descent_step(x, y, H, gd_type, inv_sigma,
                                                   step_size)
            # To update estimation by IHT or HTP.
            x = self.update_signal_estimation(x, y, H, lambda_step, gd_type, i,
                                              iter_type)
            if errors_needed:
                gener_errors[i] = Error().get_gener_error(
                    x_original, x, SIGMA_half)
            # update threshold
            if not (gd_type == constants.FAST_NEWTON_NAME
                    and i % self.fast_newton_num_gd == 0):
                lambda_step *= 0.95
            if lambda_step < final_threshold:
                lambda_step = final_threshold
        if errors_needed:
            return (x, gener_errors)
        return x

    def get_errors_by_cv(self,
                         x_original,
                         y,
                         H,
                         num_iter,
                         SIGMA_half,
                         gd_type: str,
                         iter_type: str,
                         validation_errors_needed=False):
        """
        @param x_original - in order to compute error,  we need the original/true signal.
        @param y - the observations/response.
        @param H - the design matrix.
        @param num_iter - the number of iterations.
        @param SIGMA_half - half of the covariance of design matrix.
        @param gd_type - includes gradient descent (gd), natural gd (ngd), newton method (newton).
        @param iter_type - includes iterative hard threshold (IHT), hard threshold pursuit (HTP).
        @param validation_errors_needed - If True, return validation errors map/dict.
        @return x - estimation \hat{x};
                gener_errors - generalization errors of estimation in each iteration.
        """
        #inv_sigma = pinvh(np.dot(SIGMA_half, SIGMA_half))
        inv_sigma = np.diag(1. / (np.diag(SIGMA_half)**2))
        gd_step_size = self.get_gd_step_size(H, gd_type, inv_sigma)
        # tuning lambda - cross validation
        algo_f = lambda _y, _H, _para: self.get_estimation(x_original,
                                                           _y,
                                                           _H,
                                                           num_iter,
                                                           SIGMA_half,
                                                           gd_type,
                                                           iter_type,
                                                           _para,
                                                           errors_needed=False)
        cv_obj = cv(algo_f, y, H, k=3)
        if not validation_errors_needed:
            best_lambda = cv_obj.tune_para(
                errors_needed=validation_errors_needed)  # the best threshold
        else:
            best_lambda, map_para_error = cv_obj.tune_para(
                validation_errors_needed)
        x, gener_errors = self.get_estimation(x_original,
                                              y,
                                              H,
                                              num_iter,
                                              SIGMA_half,
                                              gd_type,
                                              iter_type,
                                              best_lambda,
                                              errors_needed=True)
        if validation_errors_needed:
            return x, best_lambda, gener_errors, map_para_error
        return x, best_lambda, gener_errors