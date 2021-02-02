import tensorflow as tf
import numpy as np

from .core import Balancer


class GradientBalancer(Balancer):
    def __init__(self, group_vectors):
        self._group_vectors = np.asarray(group_vectors, dtype='float32')
        assert len(self._group_vectors.shape) == 2
        self._tf_group_vectors = tf.convert_to_tensor(self._group_vectors)
        self._n_vectors = self._group_vectors.shape[0]
        self._n_classes = self._group_vectors.shape[1]

        self._custom_alpha = tf.placeholder(dtype='float32', shape=[1, self._n_vectors])
        self._alpha = tf.Variable(tf.ones_like(self._custom_alpha))
        self._set_alpha_op = tf.assign(self._alpha, self._custom_alpha)
        self._target_distribution = tf.placeholder(dtype='float32', shape=[1, self._n_classes])

    def compile(self):
        self.__build_loss()
        assert hasattr(self, '_optimizer')
        self._minimize_op = self._optimizer.minimize(
            loss=self._loss,
            var_list=[self._alpha]
        )
        self._sess.run(tf.variables_initializer(self._optimizer.variables() + [self._alpha]))

    def __build_loss(self):
        cardinality = tf.reduce_sum(self._alpha)
        class_distribution = tf.matmul(self._alpha, self._tf_group_vectors) / cardinality

        loss = tf.reduce_mean((class_distribution - self._target_distribution)**2)
        self._loss = loss + self.__regularization()

    # noinspection PyAttributeOutsideInit
    def set_optimizer_sess(self, optimizer: tf.train.Optimizer, session: tf.Session):
        self._optimizer = optimizer
        self._sess = session

    def set_alpha(self, alpha):
        assert hasattr(self, '_sess'), 'No session is set.'
        self._sess.run(
            self._set_alpha_op,
            feed_dict={
                self._custom_alpha: np.asarray(alpha, dtype='float32').reshape(1, self._n_vectors)
            }
        )

    def add_regularization(self, init_alpha, scale=1e-3):
        init_alpha = np.asarray(init_alpha).reshape(1, self._n_vectors)
        self._reg = tf.reduce_sum((self._alpha - init_alpha)**2) * scale

    def __regularization(self):
        if hasattr(self, '_reg'):
            return self._reg
        return 0

    def balance(self, init_alpha, target_distribution, iterations=100):
        assert hasattr(self, '_optimizer'), 'Optimizer is not set.'
        init_alpha = np.asarray(init_alpha, dtype='float32').reshape(1, self._n_vectors)
        self.set_alpha(init_alpha)
        for _ in range(iterations):
            alpha, _ = self._sess.run(
                [self._alpha, self._minimize_op],
                feed_dict={
                    self._target_distribution: target_distribution
                }
            )
        return alpha


if __name__ == '__main__':
    H = np.random.randint(low=0, high=2, size=(10, 5))
    a = np.abs(np.random.randn(1, 10) * 50)
    print('initial a:', a)
    ones = np.ones((1, 5), dtype='float32')

    balancer = GradientBalancer(H)
    session = tf.Session()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
    balancer.set_optimizer_sess(optimizer, session)
    balancer.add_regularization(a, 1e-2)
    balancer.compile()
    alpha = balancer.balance(
        init_alpha=a,
        target_distribution=ones
    )
    print(alpha)
