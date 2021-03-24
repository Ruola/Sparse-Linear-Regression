import math
import numpy as np
from typing import Callable, Dict

import utils.constants as constants

class CrossValidation:
    """k fold cross validation. Used by ista_cross_validation.py
    """
    def __init__(self, algo_f: Callable, y, H, k: int):
        """
        @param algo_f - estimate the signal. 
            Inputs are response(y), design(H), tuned parameter(threshold). 
            Output the estimation.
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
        self.n = len(self.H)

    def get_vali_error(self, x, y_val, H_val, para):
        """Get the validation error.
        
        L2 norm between true value and prediction of the validation dataset.
        """
        y_pred = np.dot(H_val, x)
        return np.linalg.norm(y_val - y_pred, 2)**2 / len(y_val) / 2.

    def tune_para(self, errors_needed = False):
        """Get the best threshold/lambda by k fold cross validation.
        
        @return the best threshold as defaulted.
        If errors_needed is True, return a parameter-error map.
        """
        smallest_val_error = 1000000.
        best_para = None
        map_para_error: Dict[float, float] = dict()
        # map_para_error is a map with thres as keys and validation errors as values.
        for para in constants.TUNED_PARA_CROSS_VALI:
            val_error = 0.
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
            if best_para is None or val_error < smallest_val_error:
                smallest_val_error = val_error
                best_para = para
            map_para_error[para] = val_error
        if errors_needed:
            return best_para, map_para_error
        return best_para
