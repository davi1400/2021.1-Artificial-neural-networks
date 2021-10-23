import threading
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



class simple_perceptron_network:
    def __init__(self, epochs=None, number_of_neurons=None, learning_rate=None, activation_function=None):
        self.epochs = epochs
        self.number_of_neurons = number_of_neurons
        self.learning_rate = learning_rate
        self.activation_function = activation_function

        self.network = {}

    def add_bias(self, x):
        return concatenate([ones((x.shape[0], 1)), x], axis=1)

    def fit(self, x, y, x_val=None, y_val=None, alphas=None, batch=False, validation=True, bias=True):
        if bias:
            x = self.add_bias(x)
            if validation:
                x_val = self.add_bias(x_val)

        N, M = x.shape
        # Create all neurons
        for i in range(self.number_of_neurons):
            neuron = generic_neuron(N, M, activation_function=self.activation_function)
            self.network.update({
                'neuron-' + str(i): neuron
            })

        if batch:
            # TODO
            return self.batch_train()
        else:
            # Validation before train
            if validation:
                # pass
                self.k_fold_cross_validate(x_val, y_val, alphas)
            return self.online_train(x, y)

    def k_fold_cross_validate(self, x, y, alphas):
        N, M = x.shape
        validation_accuracys = []

        y = array(y, ndmin=2)
        for alpha in alphas:
            K = 10
            k_validation_accuracys = []
            for esimo in range(1, K + 1):
                L = int(x.shape[0] / K)
                x_train_val = (c_[x[:L * esimo - L, :].T, x[esimo * L:, :].T]).T
                x_test_val = (x[L * esimo - L:esimo * L, :])
                y_train_val = (c_[y[:L * esimo - L, :].T, y[esimo * L:, :].T]).T
                y_test_val = (y[L * esimo - L:esimo * L, :])


                classifier = simple_perceptron_network(epochs=self.epochs,
                                                       number_of_neurons=self.number_of_neurons,
                                                       learning_rate=alpha,
                                                       activation_function=self.activation_function)

                classifier.fit(x_train_val, y_train_val, validation=False, bias=False)
                y_out_val = classifier.predict(x_test_val, test=False)

                y_out_val = out_of_c_to_label(y_out_val)
                y_test_val = out_of_c_to_label(y_test_val)

                calculate_metrics = metric(y_test_val, y_out_val, types=['ACCURACY'])
                metric_results = calculate_metrics.calculate(average='micro')
                k_validation_accuracys.append(metric_results['ACCURACY'])

            validation_accuracys.append(mean(k_validation_accuracys))

        best_indice_alpha = argmax(validation_accuracys)
        self.learning_rate = alphas[best_indice_alpha]

    def one_out_c(self, u):
        y = zeros((len(u), 1))
        y[argmax(u)] = 1
        return y

    def predict(self, x, test=True):
        """

        @param test:
        @param x:
        @return:
        """
        if test:
            x = self.add_bias(x)
            outputs = zeros((x.shape[0], self.number_of_neurons))
            indice = 0
            for example in x:
                output = self.foward(example)
                y = zeros((self.number_of_neurons, 1))
                i = 0
                for neuron_key in output.keys():
                    y[i] = output[neuron_key]
                    i += 1

                outputs[indice] = y.T
                indice += 1

            return outputs
        else:
            outputs = zeros((x.shape[0], self.number_of_neurons))
            indice = 0
            for example in x:
                output = self.foward(example)
                y = zeros((self.number_of_neurons, 1))
                i = 0
                for neuron_key in output.keys():
                    y[i] = output[neuron_key]
                    i += 1

                outputs[indice] = y.T
                indice += 1

            return outputs

    def foward(self, x):
        indice = 0
        outputs = {}
        # outputs = zeros((x.shape[0], 1))
        for neuron_key in self.network.keys():
            outputs.update({
                neuron_key: self.network[neuron_key].foward(array(x, ndmin=2))
            })

        return outputs

    def backward(self, error, x):



        for neuron_key in self.network.keys():
            adjust = learning_rule_perceptron(1, error[neuron_key], array(x, ndmin=2), self.learning_rate)
            self.network[neuron_key].backward(adjust)

    def calculate_error(self, y, y_output):

        indice = 0
        error_per_neuron = {}
        for neuron_key in y_output.keys():
            error_per_neuron.update({
                neuron_key: y[indice] - y_output[neuron_key]
            })
            indice += 1

        return error_per_neuron

    def online_train(self, x, y):
        k = 0
        N, M = x.shape
        r = permutation(N)

        self.errors_per_epoch = []
        for epoch in range(self.epochs):
            y_output = self.foward(x[r[k]])
            neurons_error = self.calculate_error(array(y[r[k]], ndmin=2).T, y_output)
            # error = array(y[r[k]], ndmin=2).T - y_output

            self.backward(neurons_error, x[r[k]])
            k += 1

            if k >= r.shape[0]:
                """
                    Verificar se já passou por todos os exemplos se sim, 
                    fazer novamente randperm() e colocar o contador no 0
                """
                r = permutation(x.shape[0])
                k = 0

