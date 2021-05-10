import threading
import pandas as pd

# from numpy import
from sklearn.model_selection import KFold
from numpy.random import rand, permutation
from numpy import where, append, ones, array, zeros, mean, argmax, linspace, concatenate
from mlfwk.metrics import metric
from mlfwk.algorithms import learning_rule_perceptron, heaveside


from mlfwk.readWrite import load_mock
from multiprocessing.pool import ThreadPool
from mlfwk.models import generic_neuron
from mlfwk.utils import split_random, get_project_root, one_out_of_c



class simple_perceptron_network:
    def __init__(self, epochs=None, number_of_neurons=None, learning_rate=None, activation_function=None):
        self.epochs = epochs
        self.number_of_neurons = number_of_neurons
        self.learning_rate = learning_rate
        self.activation_function = activation_function

        self.network = {}

    def add_bias(self, x):
        return concatenate([ones((x.shape[0], 1)), x], axis=1)

    def fit(self, x, y, x_val, y_val, alphas=None, batch=False):
        x = self.add_bias(x)

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
            # self.k_fold_cross_validate(x_val, y_val, alphas)

            return self.online_train(x, y)

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
            return self.activation_function(self.foward(x))
        else:
            return self.activation_function(self.foward(x))

    def foward(self, x):
        pool = ThreadPool(processes=self.number_of_neurons)
        threads = []
        for neuron in self.network.values():
            async_result = pool.map_async(neuron.u, (array(x, ndmin=2)))
            threads.append(async_result)

        outputs = []
        for th in threads:
            outputs.append(th.get())


        return self.one_out_c(outputs)

    def backward(self, error, x):
        pool = ThreadPool(processes=self.number_of_neurons)
        threads = []
        adjust = learning_rule_perceptron(1, error.T, array(x, ndmin=2), self.learning_rate)

        for neuron in self.network.values():
            async_result = pool.map_async(neuron.backward, (adjust,))
            threads.append(async_result)

    def online_train(self, x, y):
        k = 0
        N, M = x.shape
        r = permutation(N)

        self.errors_per_epoch = []
        for epoch in range(self.epochs):
            y_output = self.foward(x[r[k]])
            error = array(y[r[k]], ndmin=2).T - y_output

            # total_error = (sum(array(y, ndmin=2) != self.predict(x, test=False)) / len(y))[0]

            # self.errors_per_epoch.append(total_error)

            self.backward(error, x[r[k]])
            k += 1

            if k >= r.shape[0]:
                """
                    Verificar se já passou por todos os exemplos se sim, 
                    fazer novamente randperm() e colocar o contador no 0
                """
                r = permutation(x.shape[0])
                k = 0



if __name__ == '__main__':
    base = load_mock(type='TRIANGLE_CLASSES')
    y_out_of_c = pd.get_dummies(base['y'])
    base = base.drop(['y'], axis=1)

    base = concatenate([base[['x1', 'x2']], y_out_of_c], axis=1)


    simple_net = simple_perceptron_network(epochs=10, number_of_neurons=3, learning_rate=0.01, activation_function='degree')

    train, test = split_random(base, train_percentage=.8)
    train, train_val = split_random(train, train_percentage=0.7)

    x_train = train[:, :2]
    y_train = train[:, 2:]

    x_train_val = train_val[:, :2]
    y_train_val = train_val[:, 2:]

    x_test = test[:, :2]
    y_test = test[:, 2:]


    validation_alphas = linspace(0.015, 0.1, 20)
    simple_net.fit(x_train, y_train, x_train_val, y_train_val, alphas=validation_alphas)
    y_out_simple_net = simple_net.predict(x_test)

    metrics_calculator = metric(list(y_test), y_out_perceptron, types=['ACCURACY', 'AUC', 'precision',
                                                                       'recall', 'f1_score', 'MCC'])
    metric_results = metrics_calculator.calculate()
    print(metric_results)
