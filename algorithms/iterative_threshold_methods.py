import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import utils.constants as constants
from utils.cross_validation import CrossValidation as cv
from utils.error import Error


class IterativeThresholdMethods:
    """Implement iterative threshold algorithms (ISTA, AdaIHT) to solve sparse linear regression.
    """
    def __init__(self, error_name = constants.GENERALIZATION_ERROR_NAME):
        """Initialize.
        
        @param error_name - prediction, generalization, and so on.
        """
        self.error_name = error_name

    def run_soft_func(self, x, y, H, lambda_para, alpha):
        """Run soft threshold function and update estimation in an iteration.
        
        soft_threshold_function(x + alpha * H.T(y-Hx), lambda)
        @param y - observed signal
        @param H - design matrix
        @param lambda_para - threshold
        @param alpha - the step size of gradient descent, alpha <= 1 / max eigen (H'H) 
        @return x - updated estimation
        """
        temp = x + alpha * np.dot(np.transpose(H), y - np.dot(H, x))
        x = temp - lambda_para * np.sign(temp)
        x[x * np.sign(temp) < 0] = 0
        return x

    def run_hard_func(self, x, y, H, lambda_step, alpha):
        """Run soft threshold function and update estimation in an iteration.
        
        hard_threshold_function(x + alpha * H.T(y-Hx), lambda)
        @param y - observed signal
        @param H - design matrix
        @param lambda_step - threshold updated in each iteration.
        @param alpha - the step size of gradient descent, alpha <= 1 / max eigen (H'H) 
        @return x - updated estimation
        """
        # hard_threshold_function(x + alpha * H.T(y-Hx), lambda)
        x = x + alpha * np.dot(np.transpose(H), y - np.dot(H, x))
        x[x < lambda_step] = 0
        return x

    def get_estimation(self,
                       y,
                       H,
                       lambda_para,
                       alpha,
                       num_iter,
                       iterative_method_type):
        """Recover the signal using ISTA or AdaIHT.

        @param y - observed signal
        @param H - design matrix
        @param lambda_para - threshold
        @param alpha - the step size of gradient descent, alpha <= 1 / max eigen (H'H) 
        @param num_iter - max number of iterations
        @param iterative_method_type - "ISTA" or "AdaIHT"
        @return x - recovered signal
        """
        x = np.zeros((constants.P))
        if iterative_method_type == constants.IHT_NAME:
            lambda_step = np.max(alpha * np.dot(np.transpose(H), y - np.dot(H, x)))
        else:  #ISTA
            lambda_step = lambda_para
        for _ in range(num_iter):
            if iterative_method_type == constants.IHT_NAME:
                x = self.run_hard_func(x, y, H, lambda_step, alpha)
                lambda_step *= 0.95  # update threshold
                if lambda_step < lambda_para:
                    lambda_step = lambda_para
            else:  #ISTA
                x = self.run_soft_func(x, y, H, lambda_step, alpha)
        return x

    def get_errors(self, y, H, lambda_para, alpha, N_iter, SIGMA_half, x_true,
                   iterative_method_type):
        """Get the estimation result and errors in each iteration.

        @param y - observed signal
        @param H - design matrix
        @param lambda_para - threshold
        @param alpha <= 1 / max eigen (H'H) is the step size of gradient descent
        @param N_iter - max number of iterations
        @param SIGMA_half - half of covariance of design matrix
        @param x_true - true signal
        @param iterative_method_type - "ISTA" or "AdaIHT"
        @return x - recovered signal; 
            errors- record errors of estimations of each iterations, 
            and the details of definition of the error are in Error class.
        """
        x = np.zeros((len(H[0])))
        if iterative_method_type == constants.IHT_NAME:
            lambda_step = np.max(alpha * np.dot(np.transpose(H), y - np.dot(H, x)))
        else:  #ISTA
            lambda_step = lambda_para
        errors = [0] * N_iter
        for i in range(N_iter):
            if iterative_method_type == constants.IHT_NAME:
                x = self.run_hard_func(x, y, H, lambda_step, alpha)
                lambda_step *= 0.95  # update threshold
                if lambda_step < lambda_para:
                    lambda_step = lambda_para
            else:  #ISTA
                x = self.run_soft_func(x, y, H, lambda_step, alpha)
            errors[i] = Error().get_error(x_true, x, self.error_name, SIGMA_half, y, H)
        return (x, errors)

    def get_errors_by_cv(self,
                         x_original,
                         y,
                         H,
                         N_iter,
                         SIGMA_half,
                         iterative_method_type,
                         validation_errors_needed=False):
        """Get the estimation result and errors in each iteration by ISTA/AdaIHT and cross validation.

        @param x_original - true signal
        @param y - observed signal
        @param H - design matrix
        @param N_iter - max number of iterations
        @param SIGMA_half - half of covariance of design matrix
        @param iterative_method_type - "ISTA" or "AdaIHT"
        @param validation_errors_needed - If True, return validation errors map/dict.
        @return x - recovered signal;
                best_lambda - best threshold;
                gener_errors - record errors of estimations of each iterations, 
                and the details of definition of the error are in Error class.
        """
        gd_step_size = 1 / 2 / np.linalg.norm(np.dot(np.transpose(H), H), 2)
        # tuning lambda - cross validation
        algo_f = lambda _y, _H, _para: self.get_estimation(
            _y,
            _H,
            _para,
            gd_step_size,
            N_iter,
            iterative_method_type)
        cv_obj = cv(algo_f,
                    y,
                    H,
                    k=5)
        if not validation_errors_needed:
            best_lambda = cv_obj.tune_para(
                errors_needed=validation_errors_needed)  # the best threshold
        else:
            best_lambda, map_para_error = cv_obj.tune_para(
                validation_errors_needed)
        x, gener_errors = self.get_errors(y, H, best_lambda, gd_step_size,
                                          N_iter, SIGMA_half, x_original,
                                          iterative_method_type)
        if validation_errors_needed:
            return x, best_lambda, gener_errors, map_para_error
        return x, best_lambda, gener_errors
