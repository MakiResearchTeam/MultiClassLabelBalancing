import pandas as pd


def load_data():
    df = pd.read_csv('uniq_hvc.csv')
    return df.to_numpy()[:, 1:]


if __name__ == '__main__':
    print(load_data())
