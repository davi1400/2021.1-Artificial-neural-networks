import numpy as np
from numpy import zeros, array, tanh
from numpy.random import randn, rand
from scipy.special import expit
from mlfwk.algorithms import heaveside


class generic_neuron:
    def __init__(self, N, M, activation_function=None, threshold_value=.5, case='threshold'):

        """

        @param N:
        @param M:
        @param activation_function:
        @param threshold_value: threslhod value used to predict in 1 or 0
        @param case: case, can be only th
        """

        self.all_possible_activations = {
            'sigmoid logistic': expit,
            'sigmoid hiperbolic': tanh,
            'degree': heaveside,
            'relu': None,
            'softmax': None,
            'linear': None
        }

        self.case = case
        self.threshold_value = threshold_value
        self.string_activation = activation_function
        self.activation = self.all_possible_activations[activation_function]
        self.weights = array(rand(M), ndmin=2).T


    def add_bias(self, x):
        """

        add the bias to the input

        @param x: input
        @return:
        """
        return np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)

    def u(self, x):
        """
        Calculate the inner dot of inputs and wheigths

        @param x:
        @return:
        """
        return x.dot(self.weights)

    def foward(self, x):

        """

        @param x:
        @return:
        """
        return self.activation(x.dot(self.weights))

    def backward(self, adjust):
        """

        @param adjust:
        @return:
        """
        self.weights += adjust

    def predict(self, x):
        """

        @param x:
        @return:
        """
        if test:
            x = self.add_bias(x)

        return self.activation(self.forward(x))

    def threshold(self, y):
        """

        @param y:
        @return:
        """
        # this case is just for one neuron
        for i in range(y.shape[0]):
            if y[i] > self.threshold_value:
                y[i] = int(1)
            else:
                y[i] = int(0) if self.string_activation == 'sigmoid logistic' else int(-1)

        return y