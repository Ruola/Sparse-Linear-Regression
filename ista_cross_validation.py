import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cross_validation import CrossValidation as cv
from error import Error
from generate_data import GenerateData


class Ista:
    """Implement iterative shrinkage threshold algorithm (ISTA) to solve sparse linear regression.
    """
    def get_estimation_by_ista(self, y, H, lambda_para, alpha, N):
        """Recover the signal using ISTA.

        @param y - observed signal
        @param H - design matrix
        @param lambda_para - threshold
        @param alpha - the step size of gradient descent, alpha <= 1 / max eigen (H'H) 
        @param N - max number of iterations
        @return x - recovered signal
        """
        x = np.zeros((len(H[0]), 1))
        for _ in range(N):
            # soft_threshold_function(x + alpha * H.T(y-Hx), lambda)
            temp = x + alpha * np.dot(np.transpose(H), y - np.dot(H, x))
            x = temp - lambda_para * np.sign(temp)
            x[x * np.sign(temp) < 0] = 0
        return x

    def get_errors_by_ista(self, y, H, lambda_para, alpha, N, SIGMA_half,
                           x_true):
        """Get the estimation result and errors in each iteration by ISTA.

        @param y - observed signal
        @param H - design matrix
        @param lambda_para - threshold
        @param alpha <= 1 / max eigen (H'H) is the step size of gradient descent
        @param N - max number of iterations
        @param SIGMA_half - half of covariance of design matrix
        @param x_true - true signal
        @return x - recovered signal; pred_errors, gener_errors- 
         record errors of estimations of each iterations, 
         and the details of definition of the error are in Error class.
        """
        x = np.zeros((len(H[0]), 1))
        pred_errors = [0] * N
        gener_errors = [0] * N
        for i in range(N):
            # soft_threshold_function(x + alpha * H.T(y-Hx), lambda)
            temp = x + alpha * np.dot(np.transpose(H), y - np.dot(H, x))
            x = temp - lambda_para * np.sign(temp)
            x[x * np.sign(temp) < 0] = 0
            pred_errors[i] = Error().get_pred_error(H, x_true, x)
            gener_errors[i] = Error().get_gener_error(x_true, x, SIGMA_half)
        return (x, pred_errors, gener_errors)

    def get_errors_by_ista_cv(self, x_original, y, H, N, SIGMA_half):
        """Get the estimation result and errors in each iteration by ISTA and cross validation.

        @param x_original - true signal
        @param y - observed signal
        @param H - design matrix
        @param N - max number of iterations
        @param SIGMA_half - half of covariance of design matrix
        @return x - recovered signal;
                best_lambda - best threshold;
                pred_errors, gener_errors - record errors of estimations of each iterations, 
                                            and the details of definition of the error are in Error class.
        """
        gd_step_size = 1 / 2 / math.ceil(max(
            np.linalg.eigh(np.dot(H.T, H))[0]))
        # tuning lambda - cross validation
        objective_f = Error().get_objective
        algo_f = lambda _y, _H, _para: self.get_estimation_by_ista(
            _y, _H, _para, gd_step_size, N)
        cv_obj = cv(objective_f,
                    algo_f,
                    y,
                    H,
                    lambda_min=0.001,
                    lambda_max=0.1,
                    params_count=100,
                    k=5)
        best_lambda = cv_obj.tune_para()  # the best threshold
        x, pred_errors, gener_errors = self.get_errors_by_ista(
            y, H, best_lambda, gd_step_size, N, SIGMA_half, x_original)
        return x, best_lambda, pred_errors, gener_errors
