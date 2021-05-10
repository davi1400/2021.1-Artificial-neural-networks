import numpy as np
from sklearn.model_selection import KFold
from numpy.random import rand, permutation
from numpy import where, append, ones, array, zeros, mean, argmin
from mlfwk.metrics import metric
from mlfwk.algorithms import learning_rule_adaline


class adaline:
    def __init__(self, learning_rate=0.1, epochs=100):

        """

        @param learning_rate:
        @param epochs:

        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def __coef__(self):
        return self.weights

    def add_bias(self, x):
        return np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)

    def fit(self, x, y, x_val, y_val, alphas=None, batch=False):
        x = self.add_bias(x)
        x_val = self.add_bias(x_val)

        N, M = x.shape
        self.weights = zeros((M, 1), dtype=np.float)

        if batch:
            # TODO
            return self.batch_train()
        else:
            # Validation before train
            self.k_fold_cross_validate(x_val, y_val, alphas)

            self.weights = zeros((M, 1), dtype=np.float)
            return self.online_train(x, y)

    def k_fold_cross_validate(self, x, y, alphas):
        N, M = x.shape
        validation_accuracys = []

        y = np.array(y, ndmin=2).T
        for alpha in alphas:
            K = 10
            k_validation_accuracys = []
            for esimo in range(1, K + 1):
                L = int(x.shape[0] / K)
                x_train_val = (np.c_[x[:L * esimo - L, :].T, x[esimo * L:, :].T]).T
                x_test_val = (x[L * esimo - L:esimo * L, :])
                y_train_val = (np.c_[y[:L * esimo - L, :].T, y[esimo * L:, :].T]).T
                y_test_val = (y[L * esimo - L:esimo * L, :])


                classifier = adaline(learning_rate=alpha, epochs=self.epochs)
                classifier.weights = zeros((M, 1))
                classifier.online_train(x_train_val, y_train_val)
                y_out_val = classifier.predict(x_test_val, test=False)

                calculate_metrics = metric(y_test_val, y_out_val, types=['MSE'])
                metric_results = calculate_metrics.calculate()
                k_validation_accuracys.append(metric_results['MSE'])

            validation_accuracys.append(mean(k_validation_accuracys))

        best_indice_alpha = argmin(validation_accuracys)
        self.learning_rate = alphas[best_indice_alpha]

    def online_train(self, x, y):
        k = 0
        N, M = x.shape
        r = permutation(N)

        self.errors_per_epoch = []
        for epoch in range(self.epochs):
            y_output = self.forward(x[r[k]])
            error = y[r[k]] - y_output

            calculate_metrics = metric(y, self.predict(x, test=False), types=['MSE'])
            metric_results = calculate_metrics.calculate()

            self.errors_per_epoch.append(metric_results['MSE'])

            self.backward(error, x[r[k]], self.learning_rate)
            k += 1

            if k >= r.shape[0]:
                """
                    Verificar se já passou por todos os exemplos se sim, 
                    fazer novamente randperm() e colocar o contador no 0
                """
                r = permutation(x.shape[0])
                k = 0

    def forward(self, x):
        """

        @param x:
        @return:
        """
        return x.dot(self.weights)

    def predict(self, x, test=True):
        """

        @param test:
        @param x:
        @return:
        """
        if test:
            x = self.add_bias(x)
            return self.forward(x)
        else:
            return self.forward(x)

    def backward(self, error, x, learning_rate):

        """

        @param error:
        @param x:
        @param learning_rate:
        @return:
        """
        try:
            self.weights += learning_rule_adaline(error, x, learning_rate)
        except Exception:
            print("here")