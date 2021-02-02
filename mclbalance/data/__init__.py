import pandas as pd
import pathlib
import os
from makiflow.augmentation.segmentation.balancing.utils import hcv_to_num


def compute_cardinalities(binary_vecs, groups):
    group_cardinalities = []
    for binary_vec in binary_vecs:
        group = hcv_to_num(binary_vec)
        group_cardinalities.append(sum(im2group == group))
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
    groups, binary_vecs = binary_vec_info[:, 0], binary_vec_info[:, 1:]

    data_path = os.path.join(data_dir_path, 'masks_hcvg.csv')
    im2group = pd.read_csv(data_path).to_numpy()[:, 1]

    binary_vecs = compute_cardinalities(binary_vecs, groups)
    return binary_vecs, groups


if __name__ == '__main__':
    print(load_data())
