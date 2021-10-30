import numpy as np
import torch
from scipy.optimize import minimize


class Balancer:
    def __init__(self, H, init_alpha):
        """
        Does minimization of the objective function using Newton method.

        Parameters
        ----------
        H : ndarray of shape [n_vectors, n_classes]
            Matrix of unique binary vectors lambda_i
        init_alpha : ndarray of shape [n_vectors]
            Vector of initial alpha parameters.
        """
        H = np.asarray(H, dtype='float64')
        init_alpha = np.asarray(init_alpha, dtype='float64').reshape(1, -1)
        # [1, n_vectors] * [n_vectors, n_classes] = [1, n_classes] == pi
        assert len(H.shape) == 2
        assert H.shape[0] == init_alpha.shape[1]
        self._H = torch.tensor(H, dtype=torch.float64)
        self._n_vectors = H.shape[0]
        self._n_classes = H.shape[1]

        self._init_alpha = torch.tensor(init_alpha, dtype=torch.float64)
        self._reg_scale = 1e-4

    def to_tensor(self, alpha):
        alpha = np.asarray(alpha, 'float64').reshape(-1)
        alpha = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)
        return alpha

    def set_reg_scale(self, reg_scale):
        self._reg_scale = reg_scale

    def loss(self, alpha):
        h_dist = torch.softmax(alpha, dim=0)
        class_frequencies = torch.matmul(h_dist, self._H)
        loss = torch.mean(-torch.log(class_frequencies))
        return loss + self._reg_scale * self.regularization(alpha)

    def regularization(self, alpha):
        ratio1 = torch.mean((torch.exp(alpha) / self._init_alpha - 1.) ** 2)
        ratio2 = torch.mean((self._init_alpha / torch.exp(alpha) - 1.) ** 2)
        return ratio1 + ratio2

    def compute_gradient(self, alpha, preprocess=True):
        if preprocess:
            alpha = np.log(alpha)

        alpha = self.to_tensor(alpha)
        loss = self.loss(alpha)
        loss.backward()
        return alpha.grad.numpy()

    def compute_hessian(self, alpha, preprocess=True):
        if preprocess:
            alpha = np.log(alpha)

        alpha = self.to_tensor(alpha)
        hessian = torch.autograd.functional.hessian(self.loss, inputs=alpha)
        return hessian.detach().numpy()

    def compute_loss(self, alpha, preprocess=True):
        if preprocess:
            alpha = np.log(alpha)

        alpha = self.to_tensor(alpha)
        loss = self.loss(alpha)
        return loss.detach().numpy()

    def balance(self, reg_scale=0.0007, method='Newton-CG'):
        init_alpha = self._init_alpha.numpy()
        init_alpha = np.log(init_alpha)
        self.set_reg_scale(reg_scale)
        loss = lambda x: self.compute_loss(x, preprocess=False)
        grad = lambda x: self.compute_gradient(x, preprocess=False)
        hess = lambda x: self.compute_hessian(x, preprocess=False)
        alpha = minimize(loss, x0=init_alpha, method=method, jac=grad, hess=hess).x
        return np.exp(alpha)
