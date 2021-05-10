import numpy as np
from sklearn.model_selection import KFold
from numpy.random import rand, permutation
from numpy import where, append, ones, array, zeros, mean, argmax
from mlfwk.metrics import metric
from mlfwk.algorithms import learning_rule_perceptron, heaveside


class perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):

        """

        @param learning_rate:
        @param epochs:
        @param test_rate:
        @param validate:

        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.g = heaveside
        self.weights = None

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


                classifier = perceptron(learning_rate=alpha, epochs=self.epochs)
                classifier.weights = zeros((M, 1))
                classifier.online_train(x_train_val, y_train_val)
                y_out_val = classifier.predict(x_test_val, test=False)

                calculate_metrics = metric(y_test_val, y_out_val, types=['ACCURACY'])
                metric_results = calculate_metrics.calculate()
                k_validation_accuracys.append(metric_results['ACCURACY'])

            validation_accuracys.append(mean(k_validation_accuracys))

        best_indice_alpha = argmax(validation_accuracys)
        self.learning_rate = alphas[best_indice_alpha]

    def online_train(self, x, y):
        k = 0
        N, M = x.shape
        r = permutation(N)

        self.errors_per_epoch = []
        for epoch in range(self.epochs):
            y_output = self.forward(x[r[k]])
            error = y[r[k]] - y_output


            total_error = (sum(np.array(y, ndmin=2) != self.predict(x, test=False))/len(y))[0]


            self.errors_per_epoch.append(total_error)
            self.backward(error, x[r[k]], self.learning_rate)
            k += 1

            if k >= r.shape[0]:
                """
                    Verificar se já passou por todos os exemplos se sim, 
                    fazer novamente randperm() e colocar o contador no 0
                """
                r = permutation(x.shape[0])
                k = 0

    def batch_train(self):
        for epoch in range(self.epochs):
            H_output = self.forward(self.X_train, weights)
            Y_output = self.predict(H_output)
            Error = array(self.Y_train, ndmin=2).T - Y_output

            accuracy = self.test(weights, self.X_test, self.Y_test, confusion_matrix=True)
            self.percents.append(accuracy)

            if abs(Error).sum() == 0:
                # print(epoch)
                break

            self.backward(weights, Error, self.X_train, self.learning_rate, H_output)

    def forward(self, x):
        """

        @param x:
        @return:
        """
        return heaveside(x.dot(self.weights))

    def predict(self, x, test=True):
        """

        @param test:
        @param x:
        @return:
        """
        if test:
            x = self.add_bias(x)
            return heaveside(self.forward(x))
        else:
            return heaveside(self.forward(x))

    def backward(self, error, x, learning_rate):

        """

        @param error:
        @param x:
        @param learning_rate:
        @return:
        """
        try:
            self.weights += learning_rule_perceptron(1, error, x, learning_rate)
        except Exception:
            print("here")