import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.io import arff
from pandas.core.frame import DataFrame
from mlfwk.utils import get_project_root
from numpy import linspace, random, cos, sin, concatenate, array, ones, pi, c_, zeros, pi, where
from numpy.random import randint, rand, seed

seed(1)


def load_base(path='', column_names=None, type=None):
    path = get_project_root() + '/mlfwk/datasets/' + path

    if type == 'csv':
        base_result = read_csv(path, names=column_names)
        return base_result
    if type == 'arff':
        base_result = arff.loadarff(path)
        base_result = DataFrame(base_result[0])
    else:
        base_result = None

    return base_result


def load_mock(type=None):
    if type == 'LOGICAL_AND':
        t = linspace(0, 2 * pi, 20)
        r = random.rand(20) / 5.0

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
        Y[20:, ] = 0 * Y[20:, ]

        return concatenate([X, Y], axis=1)

    if type == 'LOGICAL_XOR':
        t = linspace(0, 2 * pi, 50)
        r = random.rand(50) / 5.0
        data1 = c_[array((0 + r * cos(t))), array((0 + r * sin(t)))]
        data2 = c_[array((0 + r * cos(t))), array((1 + r * sin(t)))]
        data3 = c_[array((1 + r * cos(t))), array((0 + r * sin(t)))]
        data4 = c_[array((1 + r * cos(t))), array((1 + r * sin(t)))]

        X = c_[data1.T, data2.T, data3.T, data4.T].T

        Y = ones((X.shape[0], 1))
        for i in range(len(Y)):
            if i < 50:
                Y[i] = 0
            if i >= 150:
                Y[i] = 0

        return concatenate([X, Y], axis=1)

    if type == 'MOCK_SENO':
        X1 = linspace(-10, 10, 500)
        Barulho = (-1 + (1 - (-1)) * rand(500)) * 2.0
        Y = (3.0 * sin(X1) + 1) + Barulho

        return DataFrame(concatenate([X1.reshape(X1.shape[0], 1), Y.reshape(Y.shape[0], 1)], axis=1), columns=["x1", "y"])

    if type == 'TRIANGLE_CLASSES':
        t = linspace(0, 2 * pi, 50)
        r = rand(50) / 1.58
        data1 = c_[array((4 + r * cos(t))), array((4 + r * sin(t)))]
        data2 = c_[array((5 + r * cos(t))), array((2 + r * sin(t)))]
        data3 = c_[array((6 + r * cos(t))), array((4 + r * sin(t)))]

        X = c_[data1.T, data2.T, data3.T].T
        Y = zeros((150, 1))
        for i in range(50, len(Y)):
            if i < 100:
                Y[i] = 1
            else:
                Y[i] = 2

        return DataFrame(concatenate([X, Y], axis=1), columns=["x1", "x2", "y"])

    if type == 'LINEAR_REGRESSOR':
        a = randint(1, 4 + 1)
        b = randint(1, 2 + 1)

        space = linspace(1, 0.1, 100)
        noise = rand(100) / 2.0

        F = (space * a + b) + noise

        return F, space

    if type == '2D_REGRESSOR':
        a = randint(1, 4 + 1)
        b = randint(1, 2 + 1)
        c = randint(1, 2 + 1)

        space_one = linspace(1, 0.1, 100)
        space_two = linspace(1, 0.1, 100)

        noise = rand(100) / 1.5

        F = (space_one * a + b * space_two + c) + noise

        return F, space_one, space_two


if __name__ == '__main__':
    base = load_mock(type='TRIANGLE_CLASSES')

    X = base[:, :2]
    Y = base[:, 2]

    classe0 = X[where(Y == 0)[0]]
    classe1 = X[where(Y == 1)[0]]
    classe2 = X[where(Y == 2)[0]]

    plt.plot(classe0[:, 0], classe0[:, 1], '+')
    plt.plot(classe1[:, 0], classe1[:, 1], 'g*')
    plt.plot(classe2[:, 0], classe2[:, 1], 'ro')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

    # ---------------------------------------------------

    base = load_mock(type='LOGICAL_XOR')

    X = base[:, :2]
    Y = base[:, 2]

    classe0 = X[where(Y == 0)[0]]
    classe1 = X[where(Y == 1)[0]]

    plt.plot(classe0[:, 0], classe0[:, 1], '+')
    plt.plot(classe1[:, 0], classe1[:, 1], 'g*')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()