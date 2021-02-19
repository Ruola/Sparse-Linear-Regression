import math
import numpy as np
from typing import Callable


class CrossValidation:
    """k fold cross validation. Used by ista_cross_validation.py
    """
    def __init__(self, algo_f: Callable, y, H, lambda_min: float,
                 lambda_max: float, params_count: int, k: int):
        """
        @param algo_f - estimate the signal. Inputs are response(y), design(H), tuned parameter(threshold). 
        @param y - response/observation
        @param H - design matrix
        @param lambda_min - minimum of threshold/lambda chosen.
        @param lambda_max - maximum of threshold/lambda chosen.
        @param params_count - number of thresholds chosen.
        @param k - (k-fold cross validation) split the dataset into k folds and choose one as a validation set
        """
        self.k = k
        self.algo_f = algo_f  # 
        self.y = y
        self.H = H
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.params_count = params_count
        self.n = len(self.H)

    def get_vali_error(self, x, y_val, H_val, para):
        """Get the validation error.
        """
        # 1/n/2*(norm2(y-H_val*x)^2 + lambda*norm1(x))
        y_pred = np.dot(H_val, x)
        return np.linalg.norm(y_val - y_pred, 2)**2 / len(y_val) / 2.

    def tune_para(self):
        """Get the best threshold/lambda by k fold cross validation.
        
        @return the best threshold.
        """
        smallest_val_error = 1000000
        best_para = None
        for para in np.linspace(self.lambda_min, self.lambda_max,
                                self.params_count):
            val_error = 0
            for i in range(self.k):
                y_val = self.y[math.floor(i * self.n /
                                          self.k):math.floor((i + 1) * self.n /
                                                             self.k)]
                H_val = self.H[math.floor(i * self.n /
                                          self.k):math.floor((i + 1) * self.n /
                                                             self.k)]
                y_train = np.concatenate(
                    (self.y[:math.floor(i * self.n / self.k)],
                     self.y[math.floor((i + 1) * self.n / self.k):]),
                    axis=0)
                H_train = np.concatenate(
                    (self.H[:math.floor(i * self.n / self.k)],
                     self.H[math.floor((i + 1) * self.n / self.k):]),
                    axis=0)
                x = self.algo_f(y_train, H_train, para)
                val_error += self.get_vali_error(x, y_val, H_val, para)
            val_error /= self.k
            if val_error < smallest_val_error:
                smallest_val_error = val_error
                best_para = para
        # print("smallest_val_error", smallest_val_error)
        return best_para
