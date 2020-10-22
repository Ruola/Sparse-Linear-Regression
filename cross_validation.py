import math
import numpy as np

class CrossValidation:
    # input: function, k (k-fold cv), y (n*1), H (n*p matrix), lambda tuning range.
    # inputs of objective_f should be x, y_val, H_val, para. Output a number.
    # inputs of algo_f should be y_val, H_val, para. output x.
    # output: best parameter
    def __init__(self, objective_f, algo_f, y, H, lambda_min, lambda_max,params_count, k: int):
        self.k = k
        self.objective_f = objective_f
        self.algo_f = algo_f # To compute x
        self.y = y
        self.H = H
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.params_count = params_count
        self.n = len(self.H)

    def get_vali_error(self, x, y_val, H_val, para):
        # 1/n/2*(norm2(y-H_val*x)^2 + lambda*norm1(x))
        y_pred = np.dot(H_val, x)
        return np.linalg.norm(y_val - y_pred, 2) ** 2/len(y_val) / 2.

    def tune_para(self):
        smallest_val_error = 1000000
        best_para = None
        for para in np.linspace(self.lambda_min, self.lambda_max, self.params_count):
            val_error = 0
            for i in range(self.k):
                y_val = self.y[math.floor(i * self.n / self.k): math.floor((i + 1) * self.n / self.k)]
                H_val = self.H[math.floor(i * self.n / self.k): math.floor((i + 1) * self.n / self.k)]
                y_train = np.concatenate((self.y[:math.floor(i * self.n / self.k)], self.y[math.floor((i + 1) * self.n / self.k):]), axis=0)
                H_train = np.concatenate((self.H[:math.floor(i * self.n / self.k)], self.H[math.floor((i + 1) * self.n / self.k):]), axis=0)
                x = self.algo_f(y_train, H_train, para)
                val_error += self.get_vali_error(x, y_val, H_val, para)
            val_error /= self.k
            if val_error < smallest_val_error:
                smallest_val_error = val_error
                best_para = para
        print("smallest_val_error", smallest_val_error)
        return best_para
