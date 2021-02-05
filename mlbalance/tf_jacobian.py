import tensorflow as tf
from functools import partial


def flatten(tensors):
    return tf.concat([tf.reshape(tensor, [-1]) for tensor in tensors], axis=0)


def get_jac_v_op(vec_f, params, v):
    # Flatten the gradient
    vprod = tf.math.multiply(vec_f, tf.stop_gradient(v))
    Jv_op = flatten(tf.gradients(vprod, params))
    return Jv_op


def compute_jacobian(vec_f, params):
    """
    grad : tf.Tensor
        Computed gradient of a function with respect to `param`.
    """
    n = int(flatten([vec_f]).shape[0])
    grad_vp_op = partial(get_jac_v_op, vec_f, params)
    jacobian = tf.map_fn(
        fn=grad_vp_op,
        elems=tf.eye(n, n),
        dtype='float32'
    )
    return jacobian
