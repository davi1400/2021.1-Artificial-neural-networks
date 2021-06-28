import numpy as np
import threading
import pandas as pd

# from numpy import
from sklearn.model_selection import KFold
from numpy.random import rand, permutation
from numpy import where, append, ones, array, zeros, mean, argmax, linspace, concatenate, c_
from mlfwk.metrics import metric
from mlfwk.algorithms import learning_rule_multilayer_perceptron


from mlfwk.readWrite import load_mock
from multiprocessing.pool import ThreadPool
from mlfwk.models import generic_neuron
from mlfwk.utils import split_random, get_project_root, one_out_of_c, normalization, out_of_c_to_label


# TODO
class multilayer_perceptron_network:
    def __init__(self, N, M, hidden_layer_neurons=None, hidden_layers=1, activation_function=None,
                 epochs=10000, case=None, learning_rate=0.01):

        self.N = N
        self.M = M
        self.epochs = epochs
        self.hidden_layers = hidden_layers
        self.hidden_layer_neurons = hidden_layer_neurons
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.case = case
        self.network = {
            'hidden_1': {},
            'output': {}
        }

    def add_bias(self, x):
        """

        add the bias to the input

        @param x: input
        @return:
        """
        return np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)

    def fit(self, x, y, x_val=None, y_val=None, alphas=None, hidden=None, batch=False, validation=True, bias=True):
        if bias:
            x = self.add_bias(x)
            if validation:
                x_val = self.add_bias(x_val)

        N, M = x.shape
        K = y.shape[0]

        if validation:
            self.k_fold_cross_validate(x_val, y_val, alphas, hidden)

        # Create all hidden neurons
        for i in range(self.hidden_layer_neurons):
            neuron = generic_neuron(N, M, activation_function=self.activation_function)
            self.network['hidden_1'].update({
                'hidden-neuron-' + str(i): neuron
            })

        self.output_neurons = y.shape[1]
        for j in range(self.output_neurons):
            neuron = generic_neuron(N, self.hidden_layer_neurons+1, activation_function=self.activation_function)
            self.network['output'].update({
                'output-neuron-' + str(j): neuron
            })

        if batch:
            # TODO
            return self.batch_train()
        else:
            return self.online_train(x, y)

    # TODO CHANGE
    def k_fold_cross_validate(self, x, y, alphas, hidden):
        """

        @param x:
        @param y:
        @param alphas:
        @param hidden:
        @return:
        """
        K = 10
        N, M = x.shape
        validation_accuracys = np.zeros((len(hidden), len(alphas)))

        y = array(y, ndmin=2)
        i = 0
        j = 0
        for hidden_layer in hidden:
            for alpha in alphas:
                k_validation_accuracys = []
                for esimo in range(1, K + 1):
                    L = int(x.shape[0] / K)
                    x_train_val = (c_[x[:L * esimo - L, :].T, x[esimo * L:, :].T]).T
                    x_test_val = (x[L * esimo - L:esimo * L, :])
                    y_train_val = (c_[y[:L * esimo - L, :].T, y[esimo * L:, :].T]).T
                    y_test_val = (y[L * esimo - L:esimo * L, :])

                    N, M = x_train_val.shape

                    classifier = multilayer_perceptron_network(N, M,
                                                               hidden_layer_neurons=hidden_layer,
                                                               hidden_layers=1,
                                                               activation_function=self.activation_function,
                                                               epochs=1000,
                                                               case=self.case,
                                                               learning_rate=alpha)

                    classifier.fit(x_train_val, y_train_val, validation=False, bias=False)
                    y_out_val = classifier.predict(x_test_val, test=False)

                    y_out_val = out_of_c_to_label(y_out_val)
                    y_test_val = out_of_c_to_label(y_test_val)

                    calculate_metrics = metric(y_test_val, y_out_val, types=['ACCURACY'])
                    metric_results = calculate_metrics.calculate(average='micro')
                    k_validation_accuracys.append(metric_results['ACCURACY'])

                validation_accuracys[i][j] = mean(k_validation_accuracys)
                j += 1
            i += 1
            j = 0

        hidden_indice, alpha_indice = np.unravel_index(np.argmax(validation_accuracys, axis=None), validation_accuracys.shape)

        self.learning_rate = alphas[alpha_indice]
        self.hidden_layer_neurons = hidden[hidden_indice]

    def predict(self, x, test=True):
        """

        @param test:
        @param x:
        @return:
        """
        if test:
            x = self.add_bias(x)
            outputs = zeros((x.shape[0], self.output_neurons))
            indice = 0
            for example in x:
                hidden, output = self.foward(example)

                if self.output_neurons > 1:
                    y_predict = self.greater_prob(output)
                else:
                    y_predict = self.threshold(output['neuron-0'])

                y = zeros((self.output_neurons, 1))
                i = 0
                for neuron_key in y_predict.keys():
                    y[i] = y_predict[neuron_key]
                    i += 1

                outputs[indice] = y.T
                indice += 1

            return outputs
        else:
            outputs = zeros((x.shape[0], self.output_neurons))
            indice = 0
            for example in x:
                hidden, output = self.foward(example)
                y = zeros((self.output_neurons, 1))

                if self.output_neurons > 1:
                    y_predict = self.greater_prob(output)
                else:
                    y_predict = self.threshold(output['neuron-0'])

                i = 0
                for neuron_key in y_predict.keys():
                    y[i] = y_predict[neuron_key]
                    i += 1

                outputs[indice] = y.T
                indice += 1

            return outputs

    def greater_prob(self, y):
        """
        This case is only when we have more than one neuron in output layer and just one hidden layer
        @param y:
        @return:
        """
        outputs = {}
        indice_on = argmax(list(y.values()))

        # First make all zero values
        for i in range(self.output_neurons):
            outputs.update({
                'output-neuron-' + str(i): array([0.0])
            })

        # and is one the max value, or the greater probability
        outputs['output-neuron-' + str(indice_on)] = array([1.0])

        return outputs

    def foward(self, x):

        outputs = {}
        hidden = {}
        # First propagate in the hidden layers
        Hidden = np.zeros((self.hidden_layer_neurons, 1))
        counter = 0
        for hidden_neuron_key in self.network['hidden_1'].keys():
            hidden.update({
                hidden_neuron_key: self.network['hidden_1'][hidden_neuron_key].foward(array(x, ndmin=2))
            })
            Hidden[counter] = self.network['hidden_1'][hidden_neuron_key].foward(array(x, ndmin=2))
            counter += 1

        for output_neuron_key in self.network['output'].keys():
            outputs.update({
                output_neuron_key: self.network['output'][output_neuron_key].foward(array(np.c_[-1, Hidden.T], ndmin=2))
            })

        return hidden, outputs

    def backward(self, output_error, x, hidden, y):
        self.network = learning_rule_multilayer_perceptron(self.network, output_error, self.learning_rate, hidden, y, x, self.activation_function)

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
            hidden, y_output = self.foward(x[r[k]])

            if self.output_neurons > 1:
                y_predict = self.greater_prob(y_output)
            else:
                y_predict = self.threshold(y_output['neuron-0'])

            neurons_error = self.calculate_error(array(y[r[k]], ndmin=2).T,  y_output)
            # error = array(y[r[k]], ndmin=2).T - y_output
            print(neurons_error)
            self.backward(neurons_error, x[r[k]], hidden, y_output)
            k += 1

            if k >= r.shape[0]:
                """
                    Verificar se já passou por todos os exemplos se sim, 
                    fazer novamente randperm() e colocar o contador no 0
                """
                r = permutation(x.shape[0])
                k = 0



if __name__ == '__main__':
    # sem normalização

    from mlfwk.readWrite import load_mock, load_base

    # base = pd.DataFrame(load_mock(type='LOGICAL_XOR'), columns=['x1', 'x2', 'y'])
    # y_out_of_c = pd.get_dummies(base['y'])
    #
    # base = concatenate([base[['x1', 'x2']], y_out_of_c], axis=1)
    #
    #

    # carregar a base
    base = load_base(path='iris.data', type='csv')

    # normalizar a base
    base[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] = normalization(
        base[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], type='min-max')

    y_out_of_c = pd.get_dummies(base['Species'])

    base = base.drop(['Species'], axis=1)
    base = concatenate([base[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], y_out_of_c], axis=1)

    N, M = base.shape

    clf_mlp = multilayer_perceptron_network(N, M, hidden_layer_neurons=3, epochs=100, learning_rate=0.15,
                                            hidden_layers=1, activation_function='sigmoid logistic')

    train, test = split_random(base, train_percentage=.8)
    train, train_val = split_random(train, train_percentage=.8)

    x_train = train[:, :4]
    y_train = train[:, 4:]

    x_train_val = train_val[:, :4]
    y_train_val = train_val[:, 4:]

    x_test = test[:, :4]
    y_test = test[:, 4:]

    hidden_values = 3 * np.arange(1, 5)
    alpha_values = linspace(0.01, 0.1, 20)
    clf_mlp.fit(x_train, y_train, x_train_val, y_train_val, alphas=alpha_values, hidden=hidden_values, validation=False)

    y_out_simple_net = clf_mlp.predict(x_test)
    y_out = out_of_c_to_label(y_out_simple_net)
    y_test = out_of_c_to_label(y_test)

    metrics_calculator = metric(y_test, y_out, types=['ACCURACY', 'precision', 'recall', 'f1_score'])
    metric_results = metrics_calculator.calculate(average='macro')
    print(metric_results)