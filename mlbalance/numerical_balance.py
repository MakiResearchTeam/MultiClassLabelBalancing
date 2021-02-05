import numpy as np
import tensorflow as tf
from scipy.optimize import fsolve, root, OptimizeResult

from .core import Balancer
from .tf_jacobian import compute_jacobian


class NumericalBalancer(Balancer):
    REG_GAUSS = 'gaussian'
    REG_MAE = 'mae'

    def __init__(self, H, init_alpha, session, reg_scale=0.001, regularization_type=REG_GAUSS):
        """
        Parameters
        ----------
        H : ndarray of shape [n_vectors, n_classes]
            Matrix of unique binary vectors lambda_i
        init_alpha : ndarray of shape [n_vectors]
            Vector of initial alpha parameters.
        """
        H = np.asarray(H, dtype='float32')
        init_alpha = np.asarray(init_alpha, dtype='float32').reshape(1, -1)
        assert len(H.shape) == 2
        assert H.shape[0] == init_alpha.shape[1]
        self._H = tf.convert_to_tensor(H)
        self._n_vectors = H.shape[0]
        self._n_classes = H.shape[1]

        self._session = session
        self._reg_type = regularization_type

        self._init_alpha = tf.constant(init_alpha, name='init_alpha')
        self._alpha = tf.placeholder(dtype='float32', shape=[1, self._n_vectors], name='alpha')

        self._build_loss(regularization_type, reg_scale)
        self._build_gradient()
        self._build_jacobian()

    def _build_loss(self, regularization_type, reg_scale):
        class_frequencies = tf.matmul(self._alpha, self._H) / tf.reduce_sum(self._alpha)
        freq_diff = tf.reduce_mean((class_frequencies - tf.ones_like(class_frequencies))**2)

        self._loss = freq_diff + reg_scale * self._regularization(regularization_type)

    def _regularization(self, regularization_type):
        if regularization_type == NumericalBalancer.REG_GAUSS:
            squared_diff_normalized = tf.square(self._alpha - self._init_alpha)
            exp = -tf.exp(-squared_diff_normalized)
            return tf.reduce_mean(exp)

        elif regularization_type == NumericalBalancer.REG_MAE:
            squared_diff_normalized = tf.square(self._alpha - self._init_alpha)
            return tf.reduce_mean(squared_diff_normalized)

        else:
            raise ValueError(f'Unknown regularization type. Received {regularization_type}')

    def _build_gradient(self):
        self._grad = tf.gradients(self._loss, self._alpha)[0]

    def _build_jacobian(self):
        self._jac = compute_jacobian(self._grad, self._alpha)

    def compute_gradient(self, alpha):
        return self._session.run(
            self._grad,
            feed_dict={
                self._alpha: np.asarray(alpha, dtype='float32').reshape(1, -1)
            }
        ).reshape(-1)

    def compute_jacobian(self, alpha):
        return self._session.run(
            self._jac,
            feed_dict={
                self._alpha: np.asarray(alpha, dtype='float32').reshape(1, -1)
            }
        ).T

    def balance(self, init_alpha):
        return np.round(fsolve(self.compute_gradient, x0=init_alpha, xtol=1e-3, fprime=self.compute_jacobian))


if __name__ == '__main__':
    from .utils import estimate_p_classes
    from .data import load_data
    np.random.seed(1)
    data = load_data()[0].astype('float32')
    H, a = data[:, :-1], data[:, -1]
    a = a * 5
    sess = tf.Session()

    def test(reg_type, reg_scale):
        balancer = NumericalBalancer(
            H=H, init_alpha=a,
            session=sess,
            regularization_type=reg_type,
            reg_scale=reg_scale
        )
        alpha = balancer.balance(
            init_alpha=a
        )
        print('initial a:', a)
        print('balanced a:', alpha)
        print()
        print('initial frequencies:', estimate_p_classes(a, H).round(2))
        print('balanced frequencies:', estimate_p_classes(alpha, H).round(2))
        print()
        mult = (alpha / a)
        mult = np.where(mult < 1, -1 / mult, mult).round(2)
        print('multiples:', mult)
        print(balancer.compute_gradient(alpha))

    test(NumericalBalancer.REG_MAE, 0.0007)
    print('\n\n')
    test(NumericalBalancer.REG_GAUSS, 0.0023)
