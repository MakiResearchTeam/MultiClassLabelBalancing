import numpy as np


def estimate_p_classes(alpha, H):
    return alpha.dot(H) / np.sum(alpha)
