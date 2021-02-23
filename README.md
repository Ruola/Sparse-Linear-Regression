### Sparse linear regression

#### How to use
- `git clone https://github.com/Ruola/Sparse-Linear-Regression.git`
- `cd Sparse-Linear-Regression`

#### Algorithms
- Iterative soft threshold algorithm (ISTA)

- Adaptive iterative hard threshold (Ada-IHT)

- Second-order methods (gradient descent, natural gd, newton gd) + IHT / hard threshold pursuit (HTP).

### Simulations
- To compare the convergence speed of ISTA and IHT, get the change of generalization error with respect to #iterations in ISTA and IHT.
  - `python3 compare_iterative_threshold_methods.py`
- To compare gradient descent methods (gd, ngd, newton) with IHT / HTP.
  - `python3 compare_gradient_descent.py`