
import numpy as np
import pandas as pd
from numpy.random import rand, permutation
from mlfwk.metrics import metric
from scipy.special import expit
from mlfwk.metrics.metrics import accuracy_score
from mlfwk.models.KMEANS.kmeans import kmeans
from numpy import where, append, ones, array, zeros, mean, argmax, linspace, concatenate, c_, argmin


class ExtremeLearningMachines:
    def __init__(self, number_of_neurons=2, N_Classes=None, case=None):
        """

            Performs a neural network with jsut one layer(one layer of neurons + input layer), using in default sigmoid
        logistic function.

        @param epochs: Number of train epochs
        @param number_of_neurons: number of neurons in output layer
        @param learning_rate: lambda
        @param N_Classes:
        @param activation_function: the type of activate funtion in each neuron
        @param max:
        """

        self.number_of_neurons = number_of_neurons
        self.N_Classes = N_Classes
        self.wheigths = None
        self.case = case
        self.types = {
            'classification': 'ACCURACY',
            'regression': 'MSE'
        }

    def add_bias(self, x):
        """
        Add bias function

        @param x:
        @return:
        """
        return concatenate([ones((x.shape[0], 1)), x], axis=1)

    def Sigmoid(self, h):
        return expit(h)

    def fit(self, x, y, x_train_val=None, y_train_val=None, hidden=None, batch=False, validation=True, bias=True):

        """
        Fit function, training the model.

        @param hidden:
        @param x: features train set
        @param y: objectives train set
        @param x_train_val: features validation set
        @param y_train_val: objectives valition set
        @param batch: batch version, if false use the SGD
        @param validation: if true perform the k-fold cross valition with grid search
        @param bias: if true add the bias column

        @return:
        """
        if validation:
            self.k_fold_cross_validate(x_train_val, y_train_val, hidden=hidden, bias=True)


        if batch:
            # TODO
            return self.batch_train()
        else:
            if bias:
                x = self.add_bias(x)

            self.hidden_wheigths = np.random.rand(self.number_of_neurons, x.shape[1])
            return self.online_train(x, y)

    def k_fold_cross_validate(self, x, y,  hidden, bias=True):
        """

        @param x:
        @param y:
        @param hidden:
        @param bias:
        @return:
        """
        if bias:
            x = self.add_bias(x)

        K = 10
        N, M = x.shape
        validation_metrics = np.zeros((len(hidden),1))

        y = np.array(y, ndmin=2)
        if y.shape[0] == 1:
            y = y.T

        i = 0
        for hidden_layer in hidden:
            k_validation_metrics = []
            for esimo in range(1, K + 1):
                L = int(x.shape[0] / K)
                x_train_val = (c_[x[:L * esimo - L, :].T, x[esimo * L:, :].T]).T
                x_test_val = (x[L * esimo - L:esimo * L, :])
                y_train_val = (c_[y[:L * esimo - L, :].T, y[esimo * L:, :].T]).T
                y_test_val = (y[L * esimo - L:esimo * L, :])



                classifier = ExtremeLearningMachines(number_of_neurons=hidden_layer, N_Classes=self.N_Classes)

                classifier.fit(x_train_val, y_train_val, x_train_val=x_train_val, y_train_val=y_train_val, validation=False)
                y_out_val = classifier.predict(x_test_val)

                if self.case == 'classification':
                    y_test_val = classifier.predicao(y_test_val)

                calculate_metrics = metric(y_test_val, y_out_val, types=[self.types[self.case]])
                metric_results = calculate_metrics.calculate(average='micro')
                k_validation_metrics.append(metric_results[self.types[self.case]])

            validation_metrics[i] = mean(k_validation_metrics)
            i += 1


        if self.case == 'classification':
            hidden_indice, alpha_indice = np.unravel_index(np.argmax(validation_metrics, axis=None),
                                                           validation_metrics.shape)
        elif self.case == 'regression':
            hidden_indice, alpha_indice = np.unravel_index(np.argmin(validation_metrics, axis=None),
                                                           validation_metrics.shape)

        self.number_of_neurons = hidden[hidden_indice]
        print(self.number_of_neurons)

    def predicao(self, Y):
        y = np.zeros((Y.shape[0], 1))
        for j in range(Y.shape[0]):
            i = np.argmax(Y[j])
            y[j] = i
        return y

    def predict(self, x, bias=True):
        """

        @param bias: if true add bias
        @param x: features test set
        @return: predicted values
        """

        if bias:
            x = self.add_bias(x)

        H = self.foward(x)
        if self.N_Classes > 1:
            return self.greater_prob(self.Sigmoid(H.dot(self.output_wheigths)))
        else:
            return H.dot(self.output_wheigths)

    def greater_prob(self, Y):
        """
        This case is only when we have more than one neuron in output layer and just one hidden layer
        @param Y:
        @return:
        """
        y = np.zeros((Y.shape[0], 1))
        for j in range(Y.shape[0]):
            i = np.argmax(Y[j])
            y[j] = i
        return y

    def foward(self, x):
        """

        Propagate the information throgth the neuron...
        @param x:
        @return:
        """
        hidden = (self.Sigmoid(x.dot(self.hidden_wheigths.T)))
        hidden = self.add_bias(hidden)
        return hidden

    def _wheigths_adjust(self, H_train, Y):
        """

        @param H_train:
        @param Y:
        @return:
        """
        self.output_wheigths = np.linalg.pinv(H_train).dot(Y)

    def online_train(self, x, y):

        """
        Stochastic variation of train gradient descent

        @param x:
        @param y:
        @return:
        """

        H_train = self.foward(x)
        self._wheigths_adjust(H_train, y)



