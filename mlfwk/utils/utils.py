from pathlib import Path
from numpy.random import permutation
from pandas.core.frame import DataFrame


def get_project_root():
    """Returns project root folder."""
    return str(Path(__file__).parent.parent.parent)


def split_random(data_base, train_percentage=.8):
    permutation_indices = permutation(data_base.shape[0])
    print(data_base.shape[0])
    print(len(permutation_indices))
    if type(data_base) == DataFrame:
        print(permutation_indices[round(data_base.shape[0] * train_percentage)])
        train_base = data_base.iloc[permutation_indices[:round(data_base.shape[0] * train_percentage)]]
        test_base = data_base.iloc[permutation_indices[round(data_base.shape[0] * train_percentage):]]
    else:
        pass

    return train_base, test_base
