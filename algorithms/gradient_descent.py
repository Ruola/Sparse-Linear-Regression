import numpy as np
from scipy.linalg import pinvh
import utils.constants as constants
from utils.cross_validation import CrossValidation as cv
from utils.error import Error


class GradientDescent:
    """Use gradient descent (gd), natural gd or newton method with IHT/HTP to solve sparse linear regression.
    """
    def get_gd_step_size(self, H, gd_type: str, inv_sigma):
        """Compute the step size of gradient descent.
        
        @param H - the design matrix.
        @param gd_type - includes gradient descent (gd), natural gd (ngd),
                newton method (newton).
        @param inv_sigma - inverse of the design covariance matrix.
        @return
        """
        if gd_type == constants.GD_NAME:  # gradient descent
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
        if gd_type == constants.GD_NAME:
            return step_size * np.dot(np.transpose(H), y - np.dot(H, x))
        elif gd_type == constants.NGD_NAME:
            return step_size * np.dot(np.dot(inv_sigma, np.transpose(H)),
                                      y - np.dot(H, x))
        else:
            return step_size * np.dot(
                np.dot(pinvh(np.dot(np.transpose(H), H)), np.transpose(H)),
                y - np.dot(H, x))

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
        @param gd_type - includes gradient descent (gd), natural gd (ngd), newton method (newton)
        @param iter_type - includes iterative hard threshold (IHT), hard threshold pursuit (HTP)
        @param final_threshold - the lowest bound of threshold.
        @param errors_needed - If True, return generalization errors of each iteration.
        @return x - estimation \hat{x};
                gener_errors - generalization errors of estimation in each iteration
        """
        x = np.zeros((len(H[0]), 1))  # initial value of estimation
        gener_errors = [
            0
        ] * num_iter  # record generalization errors of estimation in each iteration
        inv_sigma = pinvh(np.dot(SIGMA_half, SIGMA_half))
        # Define step size of gradient descent step.
        step_size = self.get_gd_step_size(H, gd_type, inv_sigma)
        lambda_step = np.max(
            self.get_gradient_descent_step(x, y, H, gd_type, inv_sigma,
                                           step_size))  #threshold

        for i in range(num_iter):
            # Gradient descent step: gd, ngd, newton.
            x = x + self.get_gradient_descent_step(x, y, H, gd_type, inv_sigma,
                                                   step_size)
            if iter_type == constants.IHT_NAME:  # iterative hard threshold
                x[np.abs(x) < lambda_step] = 0
            else:  # HTP: hard threshold pursuit
                indices_removed = np.argwhere(
                    np.abs(x) < lambda_step
                )  # indices_removed is p * 2 dim, e.g. [[192, 1], [198, 1], [222, 1]]
                indices_removed = np.ravel(
                    np.delete(indices_removed, 1, 1)
                )  # e.g. [[192, 1], [198, 1],[222, 1]] -> [[192], [198],[222]] -> [192, 198,222]
                if len(indices_removed) > len(x):
                    # in case H'H is 0 * 0 dimension
                    H_sparse = np.delete(np.copy(H), indices_removed, 1)
                    x_temp = np.dot(
                        pinvh(np.dot(np.transpose(H_sparse), H_sparse)),
                        np.dot(np.transpose(H_sparse), y))
                    x_temp = x_temp.reshape(-1)
                    x[:, 0] = np.insert(x_temp, 0, indices_removed, 0)
            if errors_needed:
                gener_errors[i] = Error().get_gener_error(
                    x_original, x, SIGMA_half)
            # update threshold
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
        inv_sigma = inv_sigma = pinvh(np.dot(SIGMA_half, SIGMA_half))
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
        cv_obj = cv(algo_f,
                    y,
                    H,
                    lambda_min=0.0005,
                    lambda_max=0.05,
                    params_count=10,
                    k=3)
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