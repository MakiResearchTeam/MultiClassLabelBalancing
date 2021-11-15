import numpy as np
from tqdm import tqdm

from mlbalance.optim import Optimizer


class Newton(Optimizer):
    def __init__(self, it=500, hess_update_period=10, eig_correction=2.0, step_size=1.0, mu=0.5, print_period=-1, use_tqdm=False):
        """
        Parameters
        ----------
        it : int
            Number iterations for the Newton method to run.
        hess_update_period : int
            Hessian will be recomputed after every `hess_update_period` iteration.
        eig_correction : float
            This value will be added to the diagonal elements of the hessian. It improves
            the numerical stability.
        step_size : float
            The update step size.
        mu : float
            A float in range (0, 1]. The update rule for the solution is the following:
            update = compute_update(x)
            momentum = momentum * mu + update * (1 - mu)
            x = x + step_size * momentum
        print_period : int
            Information about the optimization process (the function value) will be printed
            after each `print_period` iteration. If -1, no info will be printed.
        """
        self.it = it
        self.hess_update_period = hess_update_period
        self.eig_correction = eig_correction
        self.step_size = step_size
        self.mu = mu
        self.print_period = print_period
        self.use_tqdm = use_tqdm
        self.inv_hessian_cache = None
        self.reset()

    def reset(self):
        self.inv_hessian_cache = None

    def compute_update(self, x, grad_fn, hess_fn, it: int = 0):
        if it % self.hess_update_period == 0 or self.inv_hessian_cache is None:
            hess = hess_fn(x)
            # --- Perform hessian correction
            hess = hess + np.eye(hess.shape[0], dtype=hess.dtype) * self.eig_correction
            self.inv_hessian_cache = np.linalg.inv(hess)

        grad = grad_fn(x)
        return np.dot(self.inv_hessian_cache, grad)

    def optimize(self, x0, fn, grad_fn, hess_fn):
        """
        Finds a minimum for the function `fn` using Newton method.

        Parameters
        ----------
        x0 : np.ndarray
            Initial solution.
        fn : callable
            Computes the function's value at a particular `x`.
            Takes in only one argument - the `x` vector.
        grad_fn : callable
            Computes the function's gradient at a particular `x`.
            Takes in only one argument - the `x` vector.
        hess_fn : callable
            Computes the function's hessian at a particular `x`.
            Takes in only one argument - the `x` vector.

        Returns
        -------
        np.ndarray
            A found solution.
        """
        self.reset()

        # --- Initialize momentum and perform the first iteration
        momentum = self.compute_update(x0, grad_fn, hess_fn)
        x = x0 - self.step_size * momentum

        self.print_info(0, fn(x), self.step_size * momentum)

        iterator = range(1, self.it)
        if self.use_tqdm:
            iterator = tqdm(iterator)
        for it in iterator:
            update = self.compute_update(x, grad_fn, hess_fn, it)
            momentum = momentum * self.mu + update * (1 - self.mu)
            x = x - self.step_size * momentum

            self.print_info(it, fn(x), self.step_size * momentum)
        return x

    def print_info(self, it, fn_val, delta_x):
        if it % self.print_period == 0 and self.print_period > 0:
            print(f'it={it}, fn(x)={fn_val}, abs(delta_x)={np.linalg.norm(delta_x)}')
