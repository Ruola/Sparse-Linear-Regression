import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cross_validation import CrossValidation as cv
from error import Error


class AdaIht:
    """Implement adaptive iterative hard threshold algorithm (AdaIHT) to solve sparse linear regression.
    """
    def get_estimation_by_AdaIHT(self, y, H, lambda_para, alpha, num_iter):
        """Recover the signal using AdaIHT.

        @param y - observed signal
        @param H - design matrix
        @param lambda_para - threshold
        @param alpha - the step size of gradient descent, alpha <= 1 / max eigen (H'H) 
        @param num_iter - max number of iterations
        @return x - recovered signal
        """
        lambda_step = np.max(y)
        x = np.zeros((len(H[0]), 1))
        for _ in range(num_iter):
            # hard_threshold_function(x + alpha * H.T(y-Hx), lambda)
            x = x + alpha * np.dot(np.transpose(H), y - np.dot(H, x))
            x[x < lambda_step] = 0
            lambda_step /= 1.5
            if lambda_step < lambda_para:
                lambda_step = lambda_para
        return x

    def get_errors_by_AdaIHT(self, y, H, lambda_para, alpha, num_iter,
                             SIGMA_half, x_true):
        """Get the estimation result and errors in each iteration by AdaIHT.

        @param y - observed signal
        @param H - design matrix
        @param lambda_para - lowest bound of threshold
        @param alpha <= 1 / max eigen (H'H) is the step size of gradient descent
        @param num_iter - max number of iterations
        @param SIGMA_half - half of covariance of design matrix
        @param x_true - true signal
        @return x - recovered signal; pred_errors, gener_errors- 
         record errors of estimations of each iterations, 
         and the details of definition of the error are in Error class.
        """
        lambda_step = np.max(y)
        x = np.zeros((len(H[0]), 1))
        pred_errors = [0] * num_iter
        gener_errors = [0] * num_iter
        for i in range(num_iter):
            # hard_threshold_function(x + alpha * H.T(y-Hx), lambda)
            x = x + alpha * np.dot(np.transpose(H), y - np.dot(H, x))
            x[x < lambda_step] = 0
            lambda_step *= 0.9
            if lambda_step < lambda_para:
                lambda_step = lambda_para
            pred_errors[i] = Error().get_pred_error(H, x_true, x)
            gener_errors[i] = Error().get_gener_error(x_true, x, SIGMA_half)
        return (x, pred_errors, gener_errors)

    def get_errors_by_AdaIHT_cv(self, x_original, y, H, num_iter, SIGMA_half):
        """Get the estimation result and errors in each iteration by AdaIHT and cross validation.

        @param x_original - true signal
        @param y - observed signal
        @param H - design matrix
        @param num_iter - max number of iterations
        @param SIGMA_half - half of covariance of design matrix
        @return x - recovered signal;
                best_lambda - best threshold;
                pred_errors, gener_errors - record errors of estimations of each iterations, 
                                            and the details of definition of the error are in Error class.
        """
        gd_step_size = 1 / 2. / np.linalg.norm(np.dot(np.transpose(H), H), 2)
        # tuning lambda - cross validation
        algo_f = lambda _y, _H, _para: self.get_estimation_by_AdaIHT(
            _y, _H, _para, gd_step_size, num_iter)
        cv_obj = cv(algo_f,
                    y,
                    H,
                    lambda_min=0.001,
                    lambda_max=0.1,
                    params_count=100,
                    k=5)
        best_lambda = cv_obj.tune_para()  # the best threshold
        x, pred_errors, gener_errors = self.get_errors_by_AdaIHT(
            y, H, best_lambda, gd_step_size, num_iter, SIGMA_half, x_original)
        return x, best_lambda, pred_errors, gener_errors
