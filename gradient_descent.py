import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generate_data import GenerateData
from scipy.linalg import pinvh


class GradientDescent:
    def get_objective(self, y, H, x):
        # 1/n/2*(norm2(y-H*x)^2 + lambda*norm0(x))
        return np.linalg.norm(y - np.dot(H, x), 2)**2 / len(y) / 2.

    def get_pred_error(self, H, x, x_hat):
        return np.linalg.norm(np.dot(H, x - x_hat), 2)

    def get_gener_error(self, x, x_hat, SIGMA_half):
        return np.linalg.norm(np.dot((SIGMA_half), x - x_hat), 2)

    def solve_spare_linear_regression(self, x_original, y, H, N, SIGMA_half,
                                     threshold: float, gd_type: str, iter_type: str):
        # @param x_original - in order to compute error,  we need the original/true signal.
        # @param y - the observations
        # @param H - the design matrix
        # @param N - the number of iterations
        # @param SIGMA_half - the covariance of design matrix
        # @param gd_type - 3 different gradient descent types, including gradient descent (gd), natural gradient descent (ngd), newton method (newton)
        # @param iter_type - 3 different iteration types, including iterative hard threshold (IHT), iterative soft threshold (ISTA), hard threshold pursuit (HTP)
        # @param threshold - the initial value of threshold and the default value is 200. If use cross validation, we do not need this param and can set it to be a dummy one.
        x = np.zeros((len(H[0]), 1))
        objective_funcs = [0] * N
        pred_errors = [0] * N
        gener_errors = [0] * N
        inv_sigma = pinvh(np.cov(H.T))
        if gd_type == "gd":
            step_size = 1 / 2 / np.linalg.norm(np.dot(np.transpose(H), H), 2)
        elif gd_type == "ngd":
            step_size = 1/ 2 / np.linalg.norm(
                np.dot(inv_sigma, np.dot(np.transpose(H), H)), 2)
        else:  # newton
            step_size = 1 / 2
        for i in range(N):
            if gd_type == "gd":
                # x <- x + H'(y-Hx)
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
            elif iter_type == "ISTA":  #soft threshold
                x[np.abs(x) >
                  threshold] = x[np.abs(x) > threshold] - threshold * np.sign(
                      x[np.abs(x) > threshold])
            else:  # HTP: hard threshold pursuit
                indices_removed = np.argwhere(
                    np.abs(x) <= threshold
                )  # indices_removed is p * 2 dim, e.g. [[192, 1], [198, 1]]
                indices_removed = np.ravel(
                    np.delete(indices_removed, 1, 1)
                )  # e.g. [[192, 1], [198, 1]] -> [[192], [198]] -> [192, 198]
                if len(indices_removed) > len(x):
                    # in case H'H is 0 * 0 dimension
                    H_sparse = np.delete(np.copy(H), indices_removed, 1)
                    x_temp = np.dot(
                        pinvh(np.dot(np.transpose(H_sparse), H_sparse)),
                        np.dot(np.transpose(H_sparse), y))
                    x_temp = x_temp.reshape(-1)
                    x[:, 0] = np.insert(x_temp, 0, indices_removed, 0)
            objective_funcs[i] = self.get_objective(y, H, x)
            pred_errors[i] = self.get_pred_error(H, x_original, x)
            gener_errors[i] = self.get_gener_error(x_original, x, SIGMA_half)
            if threshold > 0.05:
                threshold = threshold / 1.5  # update threshold
        return (x, threshold, objective_funcs, pred_errors, gener_errors)
