import pandas as pd
import pathlib
import os


def load_data():
    """
    Returns binary vectors and numbers of their corresponding copies.
    """
    data_dir_path = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(data_dir_path, 'uniq_hvc.csv')
    binary_vec_info = pd.read_csv(data_path).to_numpy()
    groups, binary_vec = binary_vec_info[:, 0], binary_vec_info[:, 1:]

    data_path = os.path.join(data_dir_path, 'masks_hcvg.csv')
    im2group = pd.read_csv(data_path).to_numpy()[:, 1]
    group_cardinalities = {}
    for group in groups:
        group_cardinalities[group] = sum(im2group == group)
    return binary_vec, group_cardinalities, im2group


if __name__ == '__main__':
    print(load_data())
