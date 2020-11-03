import numpy as np

def generate_data(n, p, s):
    # input: 
    # n - dimension of samples
    # p - dimension of features/predictors
    # output: x, y, H
    # y = H * x + eps
    H = np.random.randn(n, p)
    #x = np.random.rand(p, 1)
    x = 1. * np.ones((p,1))
    x[s:] = 0
    x = np.random.permutation(x)
    y = np.dot(H, x) + np.random.normal(0, 0.1, size=(n, 1))
    return (x, y, H)

def get_objective(y, H, x):
    # 1/n/2*(norm2(y-H*x)^2
    return np.linalg.norm(y - np.dot(H, x), 2) ** 2 / len(y) / 2.

def tune_alpha(y, H, w_current):
    # choose alpha
    alpha_dict = dict()
    for alpha_para in np.linspace(0.01, 0.5, 10):
        objective_results = np.zeros((p))
        for i in range(p):
            w_temp = np.copy(w_current)
            w_temp[i] += alpha_para
            objective_results[i] = get_objective(y, H, w_temp)
        i_best = np.argmin(objective_results)
        w_temp = np.copy(w_current)
        w_temp[i_best] += alpha_para
        alpha_dict[alpha_para] = get_objective(y, H, w_temp)
    alpha_para = min(alpha_dict, key = alpha_dict.get)
    return alpha_para

def foba(H, y, eps):
    # arg min ||Xw - Y||^2
    # f: R^p -> R^n
    # H = [f_1, ..., f_p]
    # output: F, w
    n = len(y)
    p = len(H[0])
    F_result = set()
    w_current = np.zeros((p))
    while (True):
        alpha_para = tune_alpha(y, H, w_current)
        print("alpha:", alpha_para)
        objective_results = np.zeros((p))
        for i in range(p):
            w_temp = np.copy(w_current)
            w_temp[i] += alpha_para
            objective_results[i] = get_objective(y, H, w_temp)
        i_best = np.argmin(objective_results)
        F_result.add(i_best)
        w_update = np.copy(w_current)
        w_update[i_best] += alpha_para
        delta = get_objective(y, H, w_current) - get_objective(y, H, w_update)
        w_current = w_update
        if (delta <= eps):
            break
        # backward
        while (True):
            objective_results = dict()
            for j in F_result:
                w_temp = np.copy(w_current)
                w_temp[j] = 0
                objective_results[j] = get_objective(y, H, w_temp)
            j_best = min(objective_results, key=objective_results.get)
            w_temp = np.copy(w_current)
            w_temp[j_best] = 0
            delta_prime = get_objective(y, H, w_temp) - get_objective(y, H, w_current)
            if (delta_prime > 0.5 * delta):
                break
            F_result.remove(j_best)
            w_current[j_best] = 0
    return (F_result, w_current)

if __name__ == "__main__":
    n, p, s = 100, 1000, 20
    x_original, y, H = generate_data(n, p, s) # n = 100, p = 1000
    print("x original", np.max(x_original))
    print("n:", n)
    print("p:", p)
    print("sparsity:", s)
    eps = 1e-6
    F_set, w = foba(H, y, eps)
    print("F: ", F_set)
    print("max of w: ", np.max(w))
    print("objective: ", get_objective(y, H, w))
