import numpy as np

import utils.constants as constants
from utils.cross_validation import CrossValidation as cv
from utils.error import Error


class FastNewton:
    """Implement the Fast Newton algorithm to solve sparse linear regression.
    
    Run several gradient descent and then cut off small coordiates once.
    """
    def get_estimation(self,
                       y,
                       H,
                       lambda_para,
                       alpha,
                       num_iter,
                       num_gd=constants.FAST_NEWTON_NUM_GD):
        """Recover the signal using Fast Newton.

        @param y - observed signal.
        @param H - design matrix.
        @param lambda_para - lower bound of threshold.
        @param alpha - the step size of gradient descent, alpha <= 1 / max eigen (H'H) .
        @param num_iter - max number of iterations.
        @param num_gd - number of gradient descent steps.
        @return x - recovered signal.
        """
        x = np.zeros((len(H[0]), 1))
        lambda_step = np.max(alpha * np.dot(np.transpose(H), y - np.dot(H, x)))
        for _ in range(num_iter):
            for i in range(num_gd):  # run several gradient descent
                x = x + alpha * np.dot(np.transpose(H), y - np.dot(H, x))
            x[x < lambda_step] = 0
            lambda_step *= 0.95  # update threshold
            if lambda_step < lambda_para:
                lambda_step = lambda_para
        return x

    def get_errors(self,
                   y,
                   H,
                   lambda_para,
                   alpha,
                   N_iter,
                   SIGMA_half,
                   x_true,
                   num_gd=constants.FAST_NEWTON_NUM_GD):
        """Get the estimation result and errors in each iteration.

        @param y - observed signal
        @param H - design matrix
        @param lambda_para - threshold
        @param alpha <= 1 / max eigen (H'H) is the step size of gradient descent
        @param N_iter - max number of iterations
        @param SIGMA_half - half of covariance of design matrix
        @param x_true - true signal
        @param num_gd - number of gradient descent steps.
        @return x - recovered signal; 
            gener_errors- record errors of estimations of each iterations, 
            and the details of definition of the error are in Error class.
        """
        x = np.zeros((len(H[0]), 1))
        lambda_step = np.max(alpha * np.dot(np.transpose(H), y - np.dot(H, x)))
        gener_errors = [0] * N_iter
        for i in range(N_iter):
            # run several gradient descent
            x = x + alpha * np.dot(np.transpose(H), y - np.dot(H, x))
            if i != 0 and i % num_gd == 0:
                x[x < lambda_step] = 0
            lambda_step *= 0.95  # update threshold
            if lambda_step < lambda_para:
                lambda_step = lambda_para
            gener_errors[i] = Error().get_gener_error(x_true, x, SIGMA_half)
        return (x, gener_errors)

    def get_errors_by_cv(self,
                         x_original,
                         y,
                         H,
                         N_iter,
                         SIGMA_half,
                         num_gd=constants.FAST_NEWTON_NUM_GD,
                         validation_errors_needed=False):
        """Get the estimation result and errors in each iteration by Fast Newton and cross validation.

        @param x_original - true signal
        @param y - observed signal
        @param H - design matrix
        @param N_iter - max number of iterations
        @param SIGMA_half - half of covariance of design matrix
        @param num_gd - number of gradient descent steps.
        @param validation_errors_needed - If True, return validation errors map/dict.
        @return x - recovered signal;
                best_lambda - best threshold;
                gener_errors - record errors of estimations of each iterations, 
                and the details of definition of the error are in Error class.
        """
        gd_step_size = 1 / 2 / np.linalg.norm(np.dot(np.transpose(H), H), 2)
        # tuning lambda - cross validation
        algo_f = lambda _y, _H, _para: self.get_estimation(
            _y, _H, _para, gd_step_size, N_iter, num_gd)
        cv_obj = cv(algo_f, y, H, k=5)
        if not validation_errors_needed:
            best_lambda = cv_obj.tune_para(
                errors_needed=validation_errors_needed)  # the best threshold
        else:
            best_lambda, map_para_error = cv_obj.tune_para(
                validation_errors_needed)
        x, gener_errors = self.get_errors(y, H, best_lambda, gd_step_size,
                                          N_iter, SIGMA_half, x_original,
                                          num_gd)
        if validation_errors_needed:
            return x, best_lambda, gener_errors, map_para_error
        return x, best_lambda, gener_errors
