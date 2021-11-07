from abc import abstractmethod

import numpy as np


class Optimizer:
    @abstractmethod
    def optimize(self, x0, fn, grad_fn, hess_fn) -> np.ndarray:
        """
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
        """
        pass
