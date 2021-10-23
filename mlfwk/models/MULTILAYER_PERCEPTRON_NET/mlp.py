import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

from numpy import c_, array, mean
from mlfwk.metrics import metric


# TODO revisar essa classe antiga que eu fiz
class MultiLayerPerceptron:
    def __init__(self, N_Padroes, N_Classes, hidden_layer_neurons=5, learning_rate=0.01, epochs=500, Regressao=False):
        self.N_Padroes = int(N_Padroes)
        self.N_Neruronios = hidden_layer_neurons
        self.N_Classes = int(N_Classes)
        self.lr = learning_rate
        self.key = Regressao
        self.epochs = epochs
        self.train_epochs_error = []
        self.done_validation = False


        if self.key:
            self.case = 'regression'
        else:
            self.case = 'classification'
        self.types = {
            'classification': 'ACCURACY',
            'regression': 'MSE'
        }

    def add_bias(self, x):
        return np.concatenate([-1*np.ones((x.shape[0], 1)), x], axis=1)

    def predicao(self, Y):
        y = np.zeros((Y.shape[0], 1))
        for j in range(Y.shape[0]):
            i = np.argmax(Y[j])
            y[j] = i
        return y

    def Sigmoid(self, h):
        return expit(h)

    def init_weigths(self):
        self.Pesos_saida = np.random.rand(self.N_Classes, self.N_Neruronios + 1)  # (cxH)
        self.Pesos_ocultos = np.random.rand(self.N_Neruronios, self.N_Padroes)  # (Hxp)

    def predict(self, X, bias=False):
        # 1. Fase de Propagação
        if bias:
            X = self.add_bias(X)

        H_Oculto = self.Sigmoid(self.Pesos_ocultos.dot(X.T))  # (Hxn)
        if self.N_Classes > 1:
            G_Saida = (
                self.Sigmoid(self.Pesos_saida.dot((np.c_[-1 * np.ones(H_Oculto.shape[1]), H_Oculto.T]).T))).T  # (cxn)
            return self.predicao(G_Saida)
        # Apenas uma classe é regressão
        elif self.N_Classes == 1 and self.key == True:
            G_Saida = (self.Pesos_saida.dot((np.c_[-1 * np.ones(H_Oculto.shape[1]), H_Oculto.T]).T)).T  # (cxn)

        return G_Saida

    def k_fold_cross_validate(self, x, y, alphas, hidden, bias=True):
        """

        @param x:
        @param y:
        @param alphas:
        @param hidden:
        @param bias:
        @return:
        """
        if bias:
            x = self.add_bias(x)

        K = 10
        N, M = x.shape
        validation_metrics = np.zeros((len(hidden), len(alphas)))

        y = np.array(y, ndmin=2)
        if y.shape[0] == 1:
            y = y.T

        i = 0
        j = 0
        for hidden_layer in hidden:
            for alpha in alphas:
                k_validation_metrics = []
                for esimo in range(1, K + 1):
                    L = int(x.shape[0] / K)
                    x_train_val = (c_[x[:L * esimo - L, :].T, x[esimo * L:, :].T]).T
                    x_test_val = (x[L * esimo - L:esimo * L, :])
                    y_train_val = (c_[y[:L * esimo - L, :].T, y[esimo * L:, :].T]).T
                    y_test_val = (y[L * esimo - L:esimo * L, :])

                    N, M = x_train_val.shape

                    classifier = MultiLayerPerceptron(M, self.N_Classes,
                                                      hidden_layer_neurons=hidden_layer,
                                                      learning_rate=alpha,
                                                      epochs=500,
                                                      Regressao=self.key)

                    classifier.fit(x_train_val, y_train_val, validation=False, bias=False)
                    y_out_val = classifier.predict(x_test_val)

                    if self.case == 'classification':
                        y_test_val = classifier.predicao(y_test_val)

                    calculate_metrics = metric(y_test_val, y_out_val, types=[self.types[self.case]])
                    metric_results = calculate_metrics.calculate(average='micro')
                    k_validation_metrics.append(metric_results[self.types[self.case]])

                validation_metrics[i][j] = mean(k_validation_metrics)
                j += 1
            i += 1
            j = 0

        if self.case == 'classification':
            hidden_indice, alpha_indice = np.unravel_index(np.argmax(validation_metrics, axis=None), validation_metrics.shape)
        elif self.case == 'regression':
            hidden_indice, alpha_indice = np.unravel_index(np.argmin(validation_metrics, axis=None), validation_metrics.shape)


        self.lr = alphas[alpha_indice]
        self.N_Neruronios = hidden[hidden_indice]
        self.done_validation = True

    def foward(self, x):
        # 1. Fase de Propagação
        hidden_output = self.Sigmoid(self.Pesos_ocultos.dot(x))  # (Hx1)
        if self.N_Classes > 1 and self.key == False:
            y_output = self.Sigmoid(self.Pesos_saida.dot((np.c_[-1, hidden_output.T]).T))  # (cx1)
        else:
            y_output = (self.Pesos_saida.dot((np.c_[-1, hidden_output.T]).T))  # (cx1)

        return hidden_output, y_output

    def backward(self, error_out, G_saida, H_Oculto, x):

        if not self.key:
            Error_Oculto = self.Pesos_saida.T.dot((G_saida * (1 - G_saida)) * error_out)  # (Hx1)
            self.Pesos_saida += self.lr * (
                ((G_saida * (1 - G_saida)) * error_out).dot((np.c_[-1, H_Oculto.T])))
        else:
            Error_Oculto = self.Pesos_saida.T.dot(1 * error_out)  # (Hx1)
            self.Pesos_saida += self.lr * (1 * error_out.dot((np.c_[-1, H_Oculto.T])))

        self.Pesos_ocultos += self.lr * (((H_Oculto * (1 - H_Oculto)) * Error_Oculto[1:, :]).dot(x.T))

    def fit(self, x_train, y_train, x_train_val=None, y_train_val=None, alphas=None, hidden=None, validation=True, bias=True):
        if validation:
            self.k_fold_cross_validate(x_train_val, y_train_val, alphas=alphas, hidden=hidden, bias=True)
        else:
            self.done_validation = True

        if bias:
            x_train = self.add_bias(x_train)

        self.init_weigths()

        for ep in range(self.epochs):
            r = np.random.permutation(x_train.shape[0])
            for k in range(len(r)):
                x = np.array((x_train[r[k]]), ndmin=2).T  # (px1)
                y = np.array((y_train[r[k]]), ndmin=2).T  # (cx1)

                hidden_output, y_output = self.foward(x)

                # 2. Propagação do erro na camada oculta
                output_error = y - y_output
                self.backward(output_error, y_output, hidden_output, x)

            if self.key and self.done_validation:
                predicted = self.predict(x_train)
                calculate_metrics = metric(predicted, y_train, types=['RMSE'])
                metric_results = calculate_metrics.calculate(average='micro')
                self.train_epochs_error.append(metric_results['RMSE'])