from pandas import read_csv
from pandas.core.frame import DataFrame
from mlfwk.utils import get_project_root
from numpy import linspace, random, cos, sin, concatenate, array, ones, pi


def load_base(path='', column_names=None, type=None):
    path = get_project_root() + '/mlfwk/datasets/' + path

    if type == 'csv':
        base_result = read_csv(path, names=column_names)
    else:
        base_result = None

    return base_result


def load_mock(type=None):
    if type == 'LOGICAL_AND':
        t = linspace(0, 2 * pi, 10)
        r = random.rand(10) / 5.0

        x1, x2 = 1 + r * cos(t), 1 + r * sin(t)
        x3, x4 = r * cos(t), r * sin(t)
        x5, x6 = 1 + r * cos(t), r * sin(t)
        x7, x8 = r * cos(t), 1 + r * sin(t)
        data1 = concatenate([x1.T, x3.T, x5.T, x7.T], axis=0)
        data2 = concatenate([x2.T, x4.T, x6.T, x8.T], axis=0)
        data1 = array(data1, ndmin=2).T
        data2 = array(data2, ndmin=2).T
        X = concatenate([data1, data2], axis=1)
        Y = ones((X.shape[0], 1))
        Y[10:, ] = 0 * Y[10:, ]

    return concatenate([X, Y], axis=1)
