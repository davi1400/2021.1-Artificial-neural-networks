import threading

import numpy as np
import pandas as pd

# from numpy import
from sklearn.model_selection import KFold
from numpy.random import rand, permutation
from numpy import where, append, ones, array, zeros, mean, argmax, linspace, concatenate, c_
from mlfwk.metrics import metric
from mlfwk.algorithms import learning_rule_perceptron


from mlfwk.readWrite import load_mock
from multiprocessing.pool import ThreadPool
from mlfwk.models import generic_neuron
from mlfwk.utils import split_random, get_project_root, one_out_of_c, normalization, out_of_c_to_label



class hyperbolic_perceptron_network:
    def __init__(self, epochs=1000, number_of_neurons=1, learning_rate=.01, activation_function='sigmoid hiperbolic'):
        """

            Performs a neural network with jsut one layer(one layer of neurons + input layer), using in default sigmoid
        hyperbolic function.

        @param epochs: Number of train epochs
        @param number_of_neurons: number of neurons in output layer
        @param learning_rate: lambda
        @param activation_function: the type of activate funtion in each neuron
        """

        self.epochs = epochs
        self.number_of_neurons = number_of_neurons
        self.learning_rate = learning_rate
        self.activation_function = activation_function

        self.network = {}

    def add_bias(self, x):
        """
        Add bias function

        @param x:
        @return:
        """
        return concatenate([ones((x.shape[0], 1)), x], axis=1)

    def fit(self, x, y, x_val=None, y_val=None, alphas=None, batch=False, validation=True, bias=True):

        """
        Fit function, training the model.

        @param x: features train set
        @param y: objectives train set
        @param x_val: features validation set
        @param y_val: objectives valition set
        @param alphas: alphas used to find the optimal hyparameter
        @param batch: batch version, if false use the SGD
        @param validation: if true perform the k-fold cross valition with grid search
        @param bias: if true add the bias column

        @return:
        """
        if bias:
            x = self.add_bias(x)
            if validation:
                x_val = self.add_bias(x_val)

        N, M = x.shape
        # Create all neurons
        for i in range(self.number_of_neurons):
            neuron = generic_neuron(N, M, activation_function=self.activation_function, threshold_value=.0)
            self.network.update({
                'neuron-' + str(i): neuron
            })

        if batch:
            # TODO
            return self.batch_train()
        else:
            # Validation before train
            if validation:
                self.k_fold_cross_validate(x_val, y_val, alphas)
            return self.online_train(x, y)

    def k_fold_cross_validate(self, x, y, alphas):
        """

        k-fold cross valition with grid search

        @param x:
        @param y:
        @param alphas:
        @return:
        """
        N, M = x.shape
        validation_accuracys = []

        y = array(y, ndmin=2).T
        for alpha in alphas:
            K = 10
            k_validation_accuracys = []
            for esimo in range(1, K + 1):
                L = int(x.shape[0] / K)
                x_train_val = (c_[x[:L * esimo - L, :].T, x[esimo * L:, :].T]).T
                x_test_val = (x[L * esimo - L:esimo * L, :])
                y_train_val = (c_[y[:L * esimo - L, :].T, y[esimo * L:, :].T]).T
                y_test_val = (y[L * esimo - L:esimo * L, :])


                classifier = hyperbolic_perceptron_network(epochs=self.epochs,
                                                       number_of_neurons=self.number_of_neurons,
                                                       learning_rate=alpha,
                                                       activation_function=self.activation_function)

                classifier.fit(x_train_val, y_train_val, validation=False, bias=False)
                y_out_val = classifier.predict(x_test_val, test=False)

                # y_out_val[np.where(y_out_val == -1)] = 0

                calculate_metrics = metric(y_test_val.tolist(), y_out_val.tolist(), types=['ACCURACY'])
                metric_results = calculate_metrics.calculate(average='macro')
                k_validation_accuracys.append(metric_results['ACCURACY'])

            validation_accuracys.append(mean(k_validation_accuracys))

        best_indice_alpha = argmax(validation_accuracys)
        self.learning_rate = alphas[best_indice_alpha]

    def predict(self, x, test=True):
        """

        @param test: if true add bias
        @param x: features test set
        @return: predicted values
        """

        if test:
            x = self.add_bias(x)

        outputs = zeros((x.shape[0], self.number_of_neurons))
        predicted_out = {}
        indice = 0
        for example in x:
            output = self.foward(example)

            if self.number_of_neurons > 1:
                pass
            else:
                # Case of perceptron, just one neuron in output layer
                output = self.threshold(output['neuron-0'])

            y = zeros((self.number_of_neurons, 1))
            i = 0
            for neuron_key in output.keys():
                y[i] = output[neuron_key]
                i += 1

            outputs[indice] = y.T
            indice += 1

        return outputs

    def threshold(self, y):
        """
        This case is only when we have one neuron in output layer and just one hidden layer
        @param y:
        @return:
        """
        outputs = {}
        for neuron_key in self.network.keys():
            outputs.update({
                neuron_key: self.network[neuron_key].threshold(array(y, ndmin=2))
            })

        return outputs

    def greater_prob(self, y):
        """
        This case is only when we have more than one neuron in output layer and just one hidden layer
        @param y:
        @return:
        """
        pass

    def foward(self, y):
        """

        Propagate the information throgth the neuron...
        @param y:
        @return:
        """
        indice = 0
        outputs = {}
        # outputs = zeros((x.shape[0], 1))
        for neuron_key in self.network.keys():
            outputs.update({
                neuron_key: self.network[neuron_key].foward(array(y, ndmin=2))
            })

        return outputs

    def backward(self, error, x, y):
        """
        Adjust the wheights

        @param error:
        @param x:
        @param y:
        @return:
        """

        for neuron_key in self.network.keys():
            y_neuron = y[neuron_key]
            adjust = learning_rule_perceptron(0.5*(1 - y_neuron), error[neuron_key], array(x, ndmin=2), self.learning_rate)
            self.network[neuron_key].backward(adjust)

    def calculate_error(self, y, y_output):

        """
        Calculate the error per neuron in the network

        @param y:
        @param y_output:
        @return:
        """

        indice = 0
        error_per_neuron = {}
        for neuron_key in y_output.keys():
            error_per_neuron.update({
                neuron_key: y[indice] - y_output[neuron_key]
            })
            indice += 1

        return error_per_neuron

    def online_train(self, x, y):

        """
        Stochastic variation of train gradient descent

        @param x:
        @param y:
        @return:
        """
        k = 0
        N, M = x.shape
        r = permutation(N)

        self.errors_per_epoch = []
        for epoch in range(self.epochs):
            y_output = self.foward(x[r[k]])

            if self.number_of_neurons > 1:
                pass
            else:
                y_predict = self.threshold(y_output['neuron-0'])

            neurons_error = self.calculate_error(array(y[r[k]], ndmin=2).T, y_predict)
            # error = array(y[r[k]], ndmin=2).T - y_output

            self.backward(neurons_error, x[r[k]], y_output)
            k += 1

            if k >= r.shape[0]:
                """
                    Verificar se já passou por todos os exemplos se sim, 
                    fazer novamente randperm() e colocar o contador no 0
                """
                r = permutation(x.shape[0])
                k = 0

