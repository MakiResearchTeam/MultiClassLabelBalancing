import numpy as np
import torch
from scipy.optimize import minimize

from mlbalance.optim import Newton, Optimizer


def newton_method() -> Optimizer:
    return Newton(
        it=512,
        hess_update_period=8,
        eig_correction=2.0,
        step_size=1.0,
        mu=0.7
    )


class BalancerV2:
    DEFAULT_REG1 = 0.007
    DEFAULT_REG2 = 0.007

    def __init__(self, H, init_alpha, device='cpu', dtype='float64'):
        """
        Does minimization of the objective function using Newton method.

        Parameters
        ----------
        H : ndarray of shape [n_vectors, n_classes]
            Matrix of unique binary vectors lambda_i
        init_alpha : ndarray of shape [n_vectors]
            Vector of initial alpha parameters.
        """
        self._device = device
        self._dtype = dtype
        self._dtype_torch = torch.float64 if dtype == 'float64' else torch.float32

        H = np.asarray(H, dtype=dtype)
        init_alpha = np.asarray(init_alpha, dtype=dtype).reshape(1, -1)
        # [1, n_vectors] * [n_vectors, n_classes] = [1, n_classes] == pi
        assert len(H.shape) == 2
        assert H.shape[0] == init_alpha.shape[1]
        self._H = torch.tensor(H, dtype=self._dtype_torch)
        self._n_vectors = H.shape[0]
        self._n_classes = H.shape[1]

        self._init_alpha_np = init_alpha
        self._init_alpha = torch.tensor(init_alpha, dtype=self._dtype_torch)
        self._reg_scale1 = 1e-4
        self._reg_scale2 = 1e-4
        self._rnd = 0.0

    def to_tensor(self, val):
        val = np.asarray(val, self._dtype).reshape(-1)
        val = torch.tensor(val, dtype=self._dtype_torch, requires_grad=True)
        val.to(self._device)
        return val

    def set_reg_scales(self, reg1: float, reg2: float):
        self._reg_scale1 = torch.tensor(reg1, dtype=self._dtype_torch)
        self._reg_scale2 = torch.tensor(reg2, dtype=self._dtype_torch)

    def set_rnd(self, rnd: float):
        # relative number of duplicates
        self._rnd = torch.tensor(rnd, dtype=self._dtype_torch)

    def set_device(self, device: str):
        self._device = device

    def loss(self, beta):
        alpha = self._init_alpha * torch.exp(beta)
        h_dist = alpha / torch.sum(alpha)
        class_frequencies = torch.matmul(h_dist, self._H)

        loss = -torch.log(class_frequencies)
        loss = torch.mean(loss)

        reg1 = self.regularization1(alpha)
        reg2 = self.regularization2(alpha)

        return loss + self._reg_scale1 * reg1 + self._reg_scale2 * reg2

    def regularization1(self, alpha):
        weights = self._init_alpha / torch.sum(self._init_alpha)
        ratio = alpha / self._init_alpha
        ratio = ratio / torch.min(ratio)
        reg = torch.sum((ratio - (1. + self._rnd)) * weights) ** 2
        return reg

    def regularization2(self, alpha):
        weights = self._init_alpha / torch.sum(self._init_alpha)
        ratio = alpha / self._init_alpha
        ratio = ratio / torch.min(ratio)
        rnd = torch.sum(ratio * weights)
        reg = torch.sum((ratio - rnd) ** 2 * weights)
        return reg

    def compute_gradient(self, params, is_alpha=True):
        params = np.asarray(params, dtype=self._dtype)
        if is_alpha:
            sigma = params / self._init_alpha_np
            beta = np.log(sigma)
        else:
            beta = params

        beta = self.to_tensor(beta)
        loss = self.loss(beta)
        loss.backward()
        return beta.grad.numpy()

    def compute_hessian(self, params, is_alpha=True):
        params = np.asarray(params, dtype=self._dtype)
        if is_alpha:
            sigma = params / self._init_alpha_np
            beta = np.log(sigma)
        else:
            beta = params

        beta = self.to_tensor(beta)
        hessian = torch.autograd.functional.hessian(self.loss, inputs=beta)
        return hessian.detach().numpy()

    def compute_loss(self, params, is_alpha=True):
        params = np.asarray(params, dtype=self._dtype)
        if is_alpha:
            sigma = params / self._init_alpha_np
            beta = np.log(sigma)
        else:
            beta = params

        beta = self.to_tensor(beta)
        loss = self.loss(beta)
        return loss.detach().numpy()

    def balance_scipy(self, rnd=0.0, reg1=DEFAULT_REG1, reg2=DEFAULT_REG2, method='Newton-CG', options=None):
        self.set_reg_scales(reg1, reg2)
        self.set_rnd(rnd)
        loss = lambda x: self.compute_loss(x, is_alpha=False)
        grad = lambda x: self.compute_gradient(x, is_alpha=False)
        hess = lambda x: self.compute_hessian(x, is_alpha=False)
        beta = minimize(loss, x0=np.zeros_like(self._init_alpha_np), method=method, jac=grad, hess=hess,
                        options=options).x
        beta -= beta.min()
        return self._init_alpha_np * np.exp(beta)

    def balance(self, rnd=0.0, reg1=DEFAULT_REG1, reg2=DEFAULT_REG2, optimizer: Optimizer = newton_method()):
        self.set_reg_scales(reg1, reg2)
        self.set_rnd(rnd)
        loss = lambda x: self.compute_loss(x, is_alpha=False)
        grad = lambda x: self.compute_gradient(x, is_alpha=False)
        hess = lambda x: self.compute_hessian(x, is_alpha=False)
        beta = optimizer.optimize(
            x0=np.zeros_like(self._init_alpha_np), fn=loss, grad_fn=grad, hess_fn=hess)
        beta -= beta.min()
        return self._init_alpha_np * np.exp(beta)
