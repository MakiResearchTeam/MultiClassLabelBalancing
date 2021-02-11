import numpy as np
import pandas as pd
import json


def estimate_p_classes(alpha, H):
    # returns pi
    return alpha.dot(H) / np.sum(alpha)


def save_cardinalities(path, card, H):
    cardinalities = card
    cardinalities = cardinalities.reshape(-1)
    cardinalities = np.round(cardinalities).astype(np.int32)
    config = {}
    for i in range(len(cardinalities)):
        config[hcv_to_num(H[i])] = cardinalities[i]
    save_json(config, path)


def hcv_to_num(bin_vec):
    num = 0
    for i in range(len(bin_vec)):
        num += int(2**i * bin_vec[i])
    return num


def save_json(d: dict, path):
    with open(path, 'w') as f:
        f.write(json.dumps(d, indent=4))


def load_json(path) -> dict:
    with open(path, 'r') as f:
        s = f.read()
    return json.loads(s)
