import pandas as pd
import pathlib
import os
from ..segmentation.balancing.utils import hcv_to_num
import numpy as np


def compute_cardinalities(binary_vecs, groups):
    group_cardinalities = []
    for binary_vec in binary_vecs:
        group = hcv_to_num(binary_vec)
        group_cardinalities.append(sum(groups == group))
    group_cardinalities = np.asarray(group_cardinalities).reshape(-1, 1)
    return np.concatenate([binary_vecs, group_cardinalities], axis=-1)


def load_data():
    """
    Returns a matrix:
    [ binary_vector, number of copies],
    and a vector of images mapped to their corresponding groups.
    """
    data_dir_path = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(data_dir_path, 'uniq_hvc.csv')
    binary_vec_info = pd.read_csv(data_path).to_numpy()
    binary_vecs = binary_vec_info[:, 1:]

    data_path = os.path.join(data_dir_path, 'masks_hcvg.csv')
    groups = pd.read_csv(data_path).to_numpy()[:, 1]

    binary_vecs = compute_cardinalities(binary_vecs, groups)
    # [[lambda, alpha]]
    return binary_vecs, groups


if __name__ == '__main__':
    print(load_data())
