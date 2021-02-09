import numpy as np
import pandas as pd


def estimate_p_classes(alpha, H):
    return alpha.dot(H) / np.sum(alpha)


def save_cardinalities(path, card, H):
    cardinalities = card
    cardinalities = cardinalities.reshape(-1)
    cardinalities = np.round(cardinalities).astype(np.int32)
    config = {}
    for i in range(len(cardinalities)):
        config[hcv_to_num(H[i])] = cardinalities[i]
    pd.DataFrame.from_dict(config, orient='index').to_csv(path)


def hcv_to_num(bin_vec):
    num = 0
    for i in range(len(bin_vec)):
        num += int(2**i * bin_vec[i])
    return num
