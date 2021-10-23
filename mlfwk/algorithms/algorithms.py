import numpy as np
from numpy import mean, array
from sklearn.metrics.pairwise import euclidean_distances


def calculate_euclidian_distance(X_example, example):
    """

    :param X_example:
    :param example:
    :return:

    """
    # dist = 0
    # for i in range(len(X_example)):
    #     dist += (X_example[i] - example[i]) ** 2
    # return np.sum((X_example.T - example)**2)
    # if example.shape == (example.shape[0],):
    #     example = np.array(example, ndmin=2)
    # if X_example.T.shape == (X_example.T.shape[0],):
    #     X_example = np.array(X_example, ndmin=2)
    #

    return euclidean_distances(X_example.reshape(1, X_example.shape[0]), example.reshape(1, example.shape[0]))


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


def get_derivate(y, func):
    """

    @param func:
    @param y
    @return:
    """
    cases = {
        'sigmoid logistic': y*(1-y),
        'linear': 1
    }

    return cases[func]


def learning_rule_multilayer_perceptron(network, output_error, learning_rate, hidden, y, x, case):
    """

    @param network:
    @param output_error:
    @param hidden:
    @param y:
    @param case:
    @param x:
    @param learning_rate
    @return:
    """
    m = len(list(hidden.values())) # number of hiddden neurons

    output_error_array = []
    hidden_wheigths = np.zeros((m+1, len(network['output'].keys())))
    output_derivates = []
    i = 0
    for output_neuron in network['output'].keys():
        output_derivates.append(get_derivate(y[output_neuron], func=case)[0])
        hidden_wheigths.T[i] = network['output'][output_neuron].weights.T
        output_error_array.append((output_error[output_neuron][0]).tolist())
        i += 1

    hidden_error = hidden_wheigths.dot(array(output_derivates, ndmin=2) * array(output_error_array, ndmin=2))[1:]
    for output_neuron in network['output'].keys():
        output_derivate = get_derivate(y[output_neuron], func=case)
        output_hidden = array(list(hidden.values()), ndmin=2).reshape(m, 1)
        adjust = array((learning_rate * (output_derivate * output_error[output_neuron] * np.c_[-1, output_hidden.T])), ndmin=2, dtype=np.float).T
        network['output'][output_neuron].backward(adjust)


    hidden_number_neuron = 0
    for hidden_neuron in network['hidden_1'].keys():
        hidden_derivate = get_derivate(hidden[hidden_neuron], func=case)
        adjust = array((learning_rate * (hidden_derivate * hidden_error[hidden_number_neuron] * x)), ndmin=2, dtype=np.float).T
        network['hidden_1'][hidden_neuron].backward(adjust)

    return network


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
