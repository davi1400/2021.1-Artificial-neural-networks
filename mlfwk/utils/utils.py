from pathlib import Path
from numpy import ndarray
from numpy.random import permutation
from pandas.core.frame import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *


def get_project_root():
    """Returns project root folder."""
    return str(Path(__file__).parent.parent.parent)


def add_diagonal(matriz, principal=True):
    if principal:
        n = len(matriz)
        assert n == len(matriz[0]), 'Matrix needs to be square'
        return sum(matriz[i][i] for i in range(n))


def calculate_confusion_matrix(y_pred, y_test):
    return confusion_matrix(y_test, y_pred)


def calculate_metrics(y_pred, y_test, metrics=None):
    cf = calculate_confusion_matrix(y_pred, y_test)

    for metric in metrics:

        if metric == 'ACCURACY':
            return 100 * add_diagonal(cf) / (1. * sum(sum(cf)))


def split_random(data_base, train_percentage=.8):
    permutation_indices = permutation(data_base.shape[0])

    if type(data_base) == DataFrame:
        train_base = data_base.iloc[permutation_indices[:round(data_base.shape[0] * train_percentage)]]
        test_base = data_base.iloc[permutation_indices[round(data_base.shape[0] * train_percentage):]]
    elif type(data_base) == ndarray:
        train_base = data_base[permutation_indices[:round(data_base.shape[0] * train_percentage)]]
        test_base = data_base[permutation_indices[round(data_base.shape[0] * train_percentage):]]
        pass

    return train_base, test_base


def normalization(x_base, type=None):
    if type == 'z-score':
        scaler = StandardScaler()
        scaler.fit(x_base)
        x_base = scaler.transform(x_base)
    elif type == 'min-max':
        x_base = (x_base - x_base.min(axis=0))/(x_base.max(axis=0)-x_base.min(axis=0))

    return x_base
