## Sparse linear regression

### How to use
- `git clone https://github.com/Ruola/Sparse-Linear-Regression.git`
- `cd Sparse-Linear-Regression`

### Algorithms
- Iterative threshold method / First-order methods (gradient descent).
  - Iterative soft threshold algorithm (ISTA)
  - Adaptive iterative hard threshold (Ada-IHT)

- Second-order methods (natural gd, newton gd) + IHT / hard threshold pursuit (HTP).

### Simulations
- To compare the convergence speed of ISTA and IHT, get the change of generalization error with respect to #iterations in ISTA and IHT.
  - `python3 compare_iterative_threshold_methods.py`.
  - Results are in `/figures/ista iht/`.
- To compare gradient descent methods (gd, ngd, newton) with IHT / HTP.
  - `python3 compare_gradient_descent.py`.
  - Results are in `/figures/second order methods/`.
- To get the change of error w.r.t. the condition number.
  - `python3 change_condition_number.py`.
  - Results are in `/figures/condition number/`.
- To get the change of exact recovery w.r.t. the signal.
  - `python3 support_recovery.py`.
  - Results are in `/figures/change signal/`.

### Unit tests

- Unit tests are in the directory `/tests/`.
- Run `python3 -m unittest tests/filename.py`, for example `python3 -m unittest tests/test_gradient_descent.py`.

### Suggestions
- The running time is very long in some simulations because of Newton and cross validation. Tmux, terminal multiplexer, is a common tool to save opening up multiple terminal sessions.
- Use Python multiprocessing to run multiple processes simultaneously.
