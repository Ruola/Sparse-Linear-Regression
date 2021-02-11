import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.linalg import pinvh

from generate_data import GenerateData
from error import Error


class GradientDescent:
    """Use gradient descent (gd), natural gd or newton method with IHT/HTP to solve sparse linear regression.
    """
    def solve_spare_linear_regression(self, x_original, y, H, N, SIGMA_half,
                                      threshold: float, gd_type: str,
                                      iter_type: str):
        """Use gradient descent (gd), natural gd or newton method with IHT/HTP to solve sparse linear regression.
        
        @param x_original - in order to compute error,  we need the original/true signal.
        @param y - the observations/response
        @param H - the design matrix
        @param N - the number of iterations
        @param SIGMA_half - half of the covariance of design matrix
        @param threshold - the initial value of threshold and the default value is 200. If use cross validation, we do not need this param and can set it to be a dummy one.
        @param gd_type - includes gradient descent (gd), natural gd (ngd), newton method (newton)
        @param iter_type - includes iterative hard threshold (IHT), hard threshold pursuit (HTP)
        @return x - estimation \hat{x}
                threshold - final threshold
                pred_errors - prediction errors of estimation in each iteration
                gener_errors - generalization errors of estimation in each iteration
        """
        x = np.zeros((len(H[0]), 1))  # initial value of estimation
        pred_errors = [
            0
        ] * N  # record prediction errors of estimation in each iteration
        gener_errors = [
            0
        ] * N  # record generalization errors of estimation in each iteration
        inv_sigma = pinvh(np.dot(SIGMA_half, SIGMA_half))
        #inv_sigma = pinvh(np.cov(H.T))
        # Define step size of gradient descent step.
        if gd_type == "gd":  # gradient descent
            step_size = 1 / 2 / np.linalg.norm(np.dot(np.transpose(H), H), 2)
        elif gd_type == "ngd":  # natural gradient descent
            step_size = 1 / 2 / np.linalg.norm(
                np.dot(inv_sigma, np.dot(np.transpose(H), H)), 2)
        else:  # newton
            step_size = 1 / 2

        for i in range(N):
            # Gradient descent step: gd, ngd, newton.
            # x <- x + alpha * M * H'(y-Hx)
            if gd_type == "gd":
                x = x + step_size * np.dot(np.transpose(H), y - np.dot(H, x))
            elif gd_type == "ngd":
                x = x + step_size * np.dot(np.dot(inv_sigma, np.transpose(H)),
                                           y - np.dot(H, x))
            else:
                x = x + step_size * np.dot(
                    np.dot(pinvh(np.dot(np.transpose(H), H)), np.transpose(H)),
                    y - np.dot(H, x))

            if iter_type == "IHT":  # iterative hard threshold
                x[np.abs(x) <= threshold] = 0
                #soft threshold
                #x[np.abs(x) > threshold] = x[np.abs(x) > threshold] - threshold * np.sign(x[np.abs(x) > threshold])
            else:  # HTP: hard threshold pursuit
                indices_removed = np.argwhere(
                    np.abs(x) <= threshold
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
            pred_errors[i] = Error().get_pred_error(H, x_original, x)
            gener_errors[i] = Error().get_gener_error(x_original, x,
                                                      SIGMA_half)
            # update threshold
            if threshold > 0.05:
                threshold = threshold / 1.5
        return (x, threshold, pred_errors, gener_errors)
