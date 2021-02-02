import pandas as pd
import pathlib
import os


def load_data():
    data_dir_path = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(data_dir_path, 'uniq_hvc.csv')
    df = pd.read_csv(data_path)
    return df.to_numpy()[:, 1:]


if __name__ == '__main__':
    print(load_data())
