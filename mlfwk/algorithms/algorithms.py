from numpy import mean


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


def calculate_perceptron_rule():
    return
