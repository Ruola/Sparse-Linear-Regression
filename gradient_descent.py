import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generate_data import GenerateData
import scipy.linalg

class GradientDescent:

    def get_objective(self, y, H, x):
        # 1/n/2*(norm2(y-H*x)^2 + lambda*norm0(x))
        return np.linalg.norm(y - np.dot(H, x), 2) ** 2 / len(y) / 2.

    def get_pred_error(self,H, x, x_hat):
        return np.linalg.norm(np.dot(H, x - x_hat), 2)

    def get_classi_error(self, x, x_hat, SIGMA):
        return np.linalg.norm(np.dot((SIGMA), x - x_hat), 2)

    def run_gradient_descent(self, x_original, y, H, N, SIGMA, type):
        x = np.zeros((len(H[0]), 1))
        objective_funcs = [0] * N
        pred_errors = [0] * N
        gener_errors = [0] * N
        lambda_para = 0.02

        inv_sigma = scipy.linalg.pinvh(np.dot(SIGMA, SIGMA))
        if type == "gd":
            # step_size = 1/np.max(np.linalg.eig(np.dot(np.transpose(H), H))[0])
            step_size = 1 /2/ np.linalg.norm(np.dot(np.transpose(H), H), 2)
        elif type == "ngd":
            #step_size = np.linalg.norm( np.dot(H, scipy.linalg.pinvh(SIGMA)), 2)**2
            step_size = 1 /2/ np.linalg.norm(np.dot(inv_sigma, np.dot(np.transpose(H), H)), 2)
        else:
            #step_size = 1 /2/ np.linalg.norm(np.dot(np.transpose(H), H), 2)
            step_size = 1 /2
        for i in range(N):
            # stopping criteria: |previous error - current error| is small
            #obj_prev = np.linalg.norm(y - np.dot(H, x), 2)
            if type == "gd":
                # x <- x - H'(y-Hx)
                x = x + step_size * np.dot(np.transpose(H), y - np.dot(H, x))
            elif type == "ngd":
                x = x + step_size * np.dot(np.dot(inv_sigma, np.transpose(H)), y - np.dot(H, x))
            else:
                x = x + step_size * np.dot(np.dot(scipy.linalg.pinvh(np.dot(np.transpose(H), H)), np.transpose(H)), y - np.dot(H, x))
            #if abs(obj_prev - np.linalg.norm(y - np.dot(H, x), 2)) < 1e-8: # stopping criteria
            #    break
            x[np.abs(x) <= lambda_para] = 0
            objective_funcs[i] = self.get_objective(y, H, x)
            pred_errors[i] = self.get_pred_error(H, x_original, x)
            gener_errors[i] = self.get_classi_error(x_original, x, SIGMA)
        return (x, objective_funcs, pred_errors,gener_errors)

