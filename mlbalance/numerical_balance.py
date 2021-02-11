import numpy as np
import tensorflow as tf
from scipy.optimize import fsolve

from .core import Balancer
from .tf_jacobian import compute_jacobian


class NumericalBalancer(Balancer):
    REG_GAUSS = 'gaussian'
    REG_MAE = 'mae'
    REG_MAE_NORM = 'mae_norm'
    REG_RATIO = 'ratio'
    REG_EXP = 'REG_EXP'
    REG_SIG = 'REG_SIG'

    def __init__(self, H, init_alpha, session, reg_scale=0.0007, regularization_type=REG_MAE):
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
        # [1, n_vectors] * [n_vectors, n_classes] = [1, n_classes] == pi
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
        class_frequencies = tf.matmul(self._alpha, self._H) / tf.reduce_sum(self._alpha)  # pi
        # freq_diff = tf.reduce_mean((class_frequencies - tf.ones_like(class_frequencies))**2)
        freq_diff = tf.reduce_mean(tf.log(class_frequencies))

        self._loss = freq_diff + reg_scale * self._regularization(regularization_type)

    def _regularization(self, regularization_type):
        if regularization_type == NumericalBalancer.REG_GAUSS:
            squared_diff_normalized = tf.square(self._alpha - self._init_alpha)
            exp = -tf.exp(-squared_diff_normalized)
            return tf.reduce_mean(exp)

        elif regularization_type == NumericalBalancer.REG_MAE:
            # .1 alpha > init_alpha
            # init_alpha = 100
            # alpha = 200
            # reg = ((400 - 100)/100)^2 = 1

            # .1 alpha < init_alpha
            # init_alpha = 100
            # alpha = 50
            # reg = ((50 - 100)/100)^2 = 1/4

            # reg = ((400 - 100))^2 / 100 =
            squared_diff_normalized = tf.square((self._alpha - self._init_alpha)) / self._init_alpha
            return tf.reduce_mean(squared_diff_normalized)

        elif regularization_type == NumericalBalancer.REG_MAE_NORM:
            diff1 = tf.square((self._alpha - self._init_alpha) / self._init_alpha)
            diff2 = tf.square((self._alpha - self._init_alpha) / self._alpha)
            return tf.reduce_mean(diff1 + diff2)

        elif regularization_type == NumericalBalancer.REG_RATIO:
            ratio_diff = tf.where(
                self._alpha > self._init_alpha,
                self._alpha / self._init_alpha - 1.,
                self._init_alpha / self._alpha - 1.
            )
            ratio_diff = tf.where(
                ratio_diff > 1.0,
                ratio_diff**2,
                tf.abs(ratio_diff)
            )
            return tf.reduce_mean(ratio_diff)
        elif regularization_type == NumericalBalancer.REG_EXP:
            exp1 = tf.exp(self._alpha / self._init_alpha)
            exp2 = tf.exp(self._init_alpha / self._alpha)
            return tf.reduce_mean((exp1 + exp2)**2)

        elif regularization_type == NumericalBalancer.REG_SIG:
            b = 3  # 3
            a = 10
            #             scale = 1e5
            x = self._alpha / self._init_alpha
            sig1 = 1 / (1 + tf.exp(-a * (x - b)))
            sig2 = 1 / (1 + tf.exp(a * (x - 1 / b)))
            return tf.reduce_sum((sig1 + sig2))

        else:
            raise ValueError(f'Unknown regularization type. Received {regularization_type}')

    def _build_gradient(self):
        # l -> min
        # grad(l) = 0
        #
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
        )

    def compute_loss(self, alpha):
        return self._session.run(
            self._loss,
            feed_dict={
                self._alpha: np.asarray(alpha, dtype='float32').reshape(1, -1)
            }
        )

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
        print('reg type:', reg_type)
        print('initial a:', a)
        print('balanced a:', alpha)
        print()
        print('initial frequencies:', estimate_p_classes(a, H).round(2))
        print('balanced frequencies:', estimate_p_classes(alpha, H).round(2))
        print()
        mult_source = (alpha / a)
        mult = np.where(mult_source < 1, -1 / mult_source, mult_source).round(2)
        print('multiples:', mult)
        print('balance_score:', np.mean(mult * mult))
        print(balancer.compute_gradient(alpha))

        scale = np.min(mult_source)
        alpha /= scale

        from .utils import save_cardinalities
        save_cardinalities(reg_type + '.csv', alpha, H)

    test(NumericalBalancer.REG_MAE, 0.0220)
    print('\n\n')
    test(NumericalBalancer.REG_GAUSS, 0.0023)
    print('\n\n')
    test(NumericalBalancer.REG_RATIO, 0.20)
    print('\n\n')
    test(NumericalBalancer.REG_MAE_NORM, 0.0000001)

