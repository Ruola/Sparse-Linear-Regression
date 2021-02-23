import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import constants
from error import Error
from generate_data import GenerateData

class Foba:

    def get_objective(self,y, H, x):
        # 1/n/2*(norm2(y-H*x)^2
        return np.linalg.norm(y - np.dot(H, x), 2) ** 2 / len(y) / 2.

    def tune_alpha(self,y, H, w_current):
        # choose alpha
        alpha_dict = dict()
        index_dict = dict() # alpha -> best index
        for alpha_para in np.linspace(0.01, 1, 20):
            objective_results = np.zeros((p))
            for i in range(p):
                w_temp = np.copy(w_current)
                w_temp[i] += alpha_para
                objective_results[i] = self.get_objective(y, H, w_temp)
            i_best = np.argmin(objective_results)
            index_dict[alpha_para] = i_best
            w_temp = np.copy(w_current)
            w_temp[i_best] += alpha_para
            alpha_dict[alpha_para] = self.get_objective(y, H, w_temp)
        alpha_para = min(alpha_dict, key = alpha_dict.get)
        return alpha_para, index_dict[alpha_para]

    def get_solution(self, H, y, F_result):
        # set coordinates in F_result into nonzero
        H_update = np.array(H)
        H_update = H_update[:, list(F_result)]
        # w = (H^T H)^(-1) H^T y
        w = np.dot(np.linalg.inv(np.dot(np.transpose(H_update), H_update)), np.dot(np.transpose(H_update), y))
        for i in range(len(H[0])):
            if i not in F_result:
                w = np.insert(w, i, 0)
        return w

    def foba(self,H, y, eps):
        # arg min ||Xw - Y||^2
        # f: R^p -> R^n
        # H = [f_1, ..., f_p]
        # output: F, w
        p = len(H[0])
        objective_record = []
        F_result = set()
        w_current = np.zeros((p))
        while (True):
            alpha_para, i_best = self.tune_alpha(y, H, w_current)
            print("alpha:", alpha_para)
            F_result.add(i_best)
            w_update = self.get_solution(H, y, F_result)
            #w_update = np.insert(w_update, list(F_result), 0, axis=0)
            objective_result_update = self.get_objective(y, H, w_update)
            delta = abs(self.get_objective(y, H, w_current) - objective_result_update)
            print("previous", self.get_objective(y, H, w_current))
            print(objective_result_update)
            w_current = w_update
            #w_current[list(F_result)] = w_update[0]
            print("max ele update", np.max(w_update))
            print("F", F_result)
            print("forward", objective_result_update)
            print("max ele", np.max(w_current))
            objective_record.append(objective_result_update)
            if (delta <= eps):
                break
            # backward
            while (True):
                objective_results = dict()
                for j in F_result:
                    w_temp = np.copy(w_current)
                    w_temp[j] = 0
                    objective_results[j] = self.get_objective(y, H, w_temp)
                if len(objective_results) < 2:
                    break
                j_best = min(objective_results, key=objective_results.get) # arg min
                w_temp = np.copy(w_current)
                w_temp[j_best] = 0
                delta_prime = self.get_objective(y, H, w_current) - self.get_objective(y, H, w_temp)
                if (delta_prime > 0.5 * delta):
                    break # break one loop
                F_result.remove(j_best)
            w_update = self.get_solution(H, y, F_result)
            z = 0
            for e in F_result:
                w_current[e] = w_update[z]
                z+=1
            objective_result_update = self.get_objective(y, H, w_current)
            print("backward", objective_result_update)
            objective_record.append(objective_result_update)
        return (F_result, w_current, objective_record)

    def run_foba(self, x_original, y, H):
        eps = 1e-1
        F_set, w, objective_record = self.foba(H, y, eps)
        print("F: ", F_set)
        print("max of w: ", np.max(w))
        print("objective: ", self.get_objective(y, H, w))
        plt.plot(objective_record)
        plt.plot([self.get_objective(y, H, x_original)] * len(objective_record)) # optimal
        plt.savefig(os.path.dirname(os.path.abspath(__file__))+"/foba")
        return objective_record

if __name__ == "__main__":
    design = "anisotropic"
    n, p, s = constants.N, constants.P, constants.S
    sigma = constants.SIGMA_NUMBER
    x_original = constants.X
    y, H = GenerateData(design).generate_data()
    print("x original", np.max(x_original))
    print("n:", n)
    print("p:", p)
    print("sparsity:", s)
    objective_funcs = Foba().run_foba(x_original, y, H)
