import numpy as np
from numpy import mean, array


def calculate_euclidian_distance(X_example, example):
    """

    :param X_example:
    :param example:
    :return:

    """
    dist = 0
    for i in range(len(X_example)):
        dist += (X_example[i] - example[i]) ** 2
    return dist


def calculate_centroid(matrix):
    """

    :param matrix:
    :return:
    """
    return mean(matrix, axis=0)


def learning_rule_perceptron(derivate_y, error, x, learning_rate):
    """

    @param derivate_y:
    @param error:
    @param x:
    @param learning_rate:
    @return:

    """
    return array((learning_rate * (derivate_y * error * x)), ndmin=2, dtype=np.float).T


def learning_rule_adaline(error, x, learning_rate):
    """

    @param derivate_y:
    @param error:
    @param x:
    @param learning_rate:
    @return:

    """
    return array((learning_rate * (error * x)), ndmin=2, dtype=np.float).T




def heaveside(y):
    """

    @param y:
    @return:
    """

    for i in range(len(y)):
        if y[i] > 0:
            y[i] = 1
        elif y[i] <= 0:
            y[i] = 0
    return y
