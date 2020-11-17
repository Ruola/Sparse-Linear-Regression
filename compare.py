import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generate_data import GenerateData
import numpy as np
from foba import Foba
from ista import Ista
from hard_threshold import HardThreshold

SIGMA = 0.1
def get_pred_error(H, x, x_hat):
    return np.linalg.norm(np.dot(H, x - x_hat), 2)

def get_classi_error(H, x, x_hat):
    return np.linalg.norm(np.dot(np.sqrt(SIGMA), x - x_hat), 2)

# compare different algorithms
steps = 50 # number of experiments
N = 100 # number of iterations in ISTA or Hard Threshold
# PARA hard_objective_matrix:
# each row represents an experiment
# and it records objection results in each iteration
ista_objective_matrix = np.zeros((steps, N))
ista_pred_errors = np.zeros((steps))
ista_classi_errors = np.zeros((steps))
hard_objective_matrix = np.zeros((steps, N))
hard_thres_pred_errors = np.zeros((steps))
hard_thres_classi_errors = np.zeros((steps))
for i in range(steps):
    x_original, y, H = GenerateData().generate_data(100, 100, 20, SIGMA) # n = 100, p = 1000
    x_ista, ista_objectives = Ista().run_ista(x_original, y, H, N)
    ista_objective_matrix[i] = ista_objectives
    ista_pred_errors[i] = get_pred_error(H, x_original, x_ista)
    ista_classi_errors[i] = get_classi_error(H, x_original, x_ista)
    x_hard_thres, hard_thres_objectives = HardThreshold().run_hard_threshold(x_original, y, H,N)
    hard_objective_matrix[i] = hard_thres_objectives
    hard_thres_pred_errors[i] = get_pred_error(H, x_original, x_hard_thres)
    hard_thres_classi_errors[i] = get_classi_error(H, x_original, x_hard_thres)
ista_result = np.mean(ista_objective_matrix, axis=0)
plt.plot(ista_result) # optimal
hard_thres_result = np.mean(hard_objective_matrix, axis=0)
plt.plot(hard_thres_result) # optimal
plt.savefig(os.path.dirname(os.path.abspath(__file__))+"/compare")
print("ISTA prediction errors: ")
print(ista_pred_errors)
print("ISTA classification errors: ")
print(ista_classi_errors)
print("Hard Threshold prediction errors: ")
print(hard_thres_pred_errors)
print("Hard Threshold classification errors: ")
print(hard_thres_classi_errors)
