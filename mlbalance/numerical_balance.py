import numpy as np
import torch
from scipy.optimize import minimize

from .optim import Newton, Optimizer


def newton_method() -> Optimizer:
    return Newton(
        it=500,
        hess_update_period=10,
        eig_correction=2.0,
        step_size=1.0,
        mu=0.7
    )


class Balancer:
    DEFAULT_REG_SCALE = 0.007

    def __init__(self, H, init_alpha, eps=0.0, device='cpu', dtype='float64', vectorize_hess=False):
        """
        Implements the loss function from the paper and provides methods for minimizing it.

        Parameters
        ----------
        H : ndarray of shape [n_vectors, n_classes]
            Matrix of unique binary vectors lambda_i
        init_alpha : ndarray of shape [n_vectors]
            Vector of initial alpha parameters.
        eps : float
            Being added to class frequencies to avoid NaN values in logarithm
            when some frequencies equal to zero.
        """
        self._device = device
        self._dtype = dtype
        self._dtype_torch = torch.float64 if dtype == 'float64' else torch.float32
        self._vectorize_hess = vectorize_hess

        H = np.asarray(H, dtype=dtype)
        init_alpha = np.asarray(init_alpha, dtype=dtype).reshape(1, -1)
        # [1, n_vectors] * [n_vectors, n_classes] = [1, n_classes] == pi
        assert len(H.shape) == 2
        assert H.shape[0] == init_alpha.shape[1]
        self._H = torch.tensor(H, dtype=self._dtype_torch, device=self._device)
        self._n_vectors = H.shape[0]
        self._n_classes = H.shape[1]

        self._init_alpha_np = init_alpha
        self._init_alpha = torch.tensor(init_alpha, dtype=self._dtype_torch, device=self._device)
        self._reg_scale = torch.tensor(1e-4, dtype=self._dtype_torch, device=self._device)

        self._eps = torch.tensor(eps, dtype=self._dtype_torch, device=self._device)

    def to_tensor(self, val):
        val = np.asarray(val, self._dtype).reshape(-1)
        val = torch.tensor(val, dtype=self._dtype_torch, requires_grad=True, device=self._device)
        return val

    def set_reg_scale(self, reg_scale):
        self._reg_scale = torch.tensor(reg_scale, dtype=self._dtype_torch, device=self._device)

    def set_device(self, device):
        self._device = device
        self._reg_scale = self._reg_scale.to(device)
        self._init_alpha = self._init_alpha.to(device)
        self._eps = self._eps.to(device)
        self._H = self._H.to(device)

    def loss(self, beta):
        alpha = self._init_alpha * torch.exp(beta)
        h_dist = alpha / torch.sum(alpha)
        class_frequencies = torch.matmul(h_dist, self._H)

        weights = 1. / (class_frequencies + self._eps) * torch.sign(class_frequencies)
        loss = weights * -torch.log(class_frequencies + self._eps) / torch.sum(weights)
        loss = torch.sum(loss)

        return loss + self._reg_scale * self.regularization(alpha)

    def regularization(self, alpha):
        weights = self._init_alpha / torch.sum(self._init_alpha)
        ratio = alpha / (self._init_alpha + self._eps)
        ratio = ratio / torch.min(ratio)
        reg = torch.sum((ratio - 1.) ** 2 * weights)
        return reg

    def compute_gradient(self, params, is_alpha=True):
        params = np.asarray(params, dtype=self._dtype)
        if is_alpha:
            sigma = params / self._init_alpha_np
            beta = np.log(sigma + self._eps.cpu().numpy())
        else:
            beta = params

        beta = self.to_tensor(beta)
        loss = self.loss(beta)
        loss.backward()
        return beta.grad.cpu().numpy()

    def compute_hessian(self, params, is_alpha=True):
        params = np.asarray(params, dtype=self._dtype)
        if is_alpha:
            sigma = params / self._init_alpha_np
            beta = np.log(sigma + self._eps.cpu().numpy())
        else:
            beta = params

        beta = self.to_tensor(beta)
        hessian = torch.autograd.functional.hessian(self.loss, inputs=beta, vectorize=self._vectorize_hess)
        return hessian.cpu().detach().numpy()

    def compute_loss(self, params, is_alpha=True):
        params = np.asarray(params, dtype=self._dtype)
        if is_alpha:
            sigma = params / self._init_alpha_np
            beta = np.log(sigma + self._eps.cpu().numpy())
        else:
            beta = params

        beta = self.to_tensor(beta)
        loss = self.loss(beta)
        return loss.cpu().detach().numpy()

    def balance_scipy(self, reg_scale=DEFAULT_REG_SCALE, method='Newton-CG', options=None):
        self.set_reg_scale(reg_scale)
        loss = lambda x: self.compute_loss(x, is_alpha=False)
        grad = lambda x: self.compute_gradient(x, is_alpha=False)
        hess = lambda x: self.compute_hessian(x, is_alpha=False)
        beta = minimize(loss, x0=np.zeros_like(self._init_alpha_np), method=method, jac=grad, hess=hess,
                        options=options).x
        beta -= beta.min()
        return self._init_alpha_np * np.exp(beta)

    def balance(self, optimizer: Optimizer = newton_method(), reg_scale=DEFAULT_REG_SCALE):
        self.set_reg_scale(reg_scale)
        loss = lambda x: self.compute_loss(x, is_alpha=False)
        grad = lambda x: self.compute_gradient(x, is_alpha=False)
        hess = lambda x: self.compute_hessian(x, is_alpha=False)
        beta = optimizer.optimize(
            x0=np.zeros_like(self._init_alpha_np), fn=loss, grad_fn=grad, hess_fn=hess)
        beta -= beta.min()
        return self._init_alpha_np * np.exp(beta)
