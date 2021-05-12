from numpy import zeros, array
from numpy.random import randn, rand
from scipy.special import expit
from mlfwk.algorithms import heaveside


class generic_neuron:
    def __init__(self, N, M, activation_function=None):
        self.all_possible_activations = {
            'sigmoid logistic': expit,
            'degree': heaveside
        }
        self.activation = self.all_possible_activations[activation_function]
        self.weights = array(rand(M), ndmin=2).T


    def add_bias(self, x):
        return np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)

    def u(self, x):
        return x.dot(self.weights)

    def foward(self, x):
        return self.activation(x.dot(self.weights))

    def backward(self, adjust):
        self.weights += adjust

    def predict(self, x):

        if test:
            x = self.add_bias(x)
            return heaveside(self.forward(x))
        else:
            return heaveside(self.forward(x))