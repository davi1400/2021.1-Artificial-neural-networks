import numpy as np
from numpy import zeros, where, ndarray, array, sqrt
from sklearn.metrics import *
from mlfwk.utils import calculate_confusion_matrix


class metric:
    def __init__(self, real_outputs, predicted_outputs, types):
        """

        :param real_outputs:
        :param predicted_outputs:
        :param types:

        """
        self.y_pred = predicted_outputs
        self.y_real = real_outputs
        self.types = types

    def calculate(self, average=None):
        result = {}
        for metric in self.types:
            result.update({metric: self.one_by_one(metric, self.y_real, self.y_pred, average)})

        return result

    def accuracy(self, y_true, y_pred, average):
        return accuracy_score(y_true, y_pred)
        # return (y_pred == y_true).sum() / (1.0 * len(y_true))

    def precision(self, y_true, y_pred, average):
        return precision_score(y_true, y_pred, average=average)

    def recall(self, y_true, y_pred, average):
        return recall_score(y_true, y_pred, average=average)

    def f1(self, y_true, y_pred, average):
        precision = self.precision(y_true, y_pred, average=average)
        recall = self.recall(y_true, y_pred, average=average)

        try:
            if np.isnan((2*precision*recall)/(precision+recall)):
                return 0
            else:
                return (2*precision*recall)/(precision+recall)
        except Exception:
            print("error")

    def mse(self, y_true, y_pred, average=None):
        return mean_squared_error(y_true, y_pred)

    def rmse(self, y_true, y_pred, average=None):
        return sqrt(self.mse(y_true, y_pred))

    def r_two(self, y_true, y_pred, average=None):
        return r2_score(y_true, y_pred)

    def one_by_one(self, type, y_true, y_pred, average='binary'):

        # o AUC e MCC estão sendo calculados com o sklearn, TODO -> criar função para calculalos na mão
        func_possibles = {
            'ACCURACY': self.accuracy,
            'AUC': roc_auc_score,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1,
            'MCC': matthews_corrcoef,
            'MSE': self.mse,
            'RMSE': self.rmse,
            'R2': self.r_two
        }

        return func_possibles[type](y_true, y_pred, average)

    @staticmethod
    def confusion_matrix(y_true, y_pred, labels):
        return confusion_matrix(y_true, y_pred, labels=labels)

    @staticmethod
    def confusion_matrix_binary(y_true, y_pred):
        """

        :param y_true:
        :param y_pred:
        :return:
                        Previsto
                       1       0
                    1
        Real        0

        """
        cf = zeros((2, 2))
        if type(y_true) != ndarray:
            y_true = array(y_true, ndmin=2).T

        if type(y_pred) != ndarray:
            y_pred = array(y_pred, ndmin=2).T


        indices_one_exepcted = where(y_true == 1.0)[0]
        indices_zero_exepcted = where(y_true == 0.0)[0]

        # Esperado é 1 e a predição deu 1
        cf[0][0] = len(where(y_pred[indices_one_exepcted] == y_true[indices_one_exepcted])[0])

        # Esperado é 1 e a predição deu 0, falso negativo
        cf[0][1] = len(where(y_pred[indices_one_exepcted] != y_true[indices_one_exepcted])[0])


        # Esperado é 0 e a predição deu 1, falso positivo
        cf[1][0] = len(where(y_pred[indices_zero_exepcted] != y_true[indices_zero_exepcted])[0])

        # Esperado é 0 e a predição deu 0
        cf[1][1] = len(where(y_pred[indices_zero_exepcted] == y_true[indices_zero_exepcted])[0])

        return cf


if __name__ == '__main__':
    from numpy import ones

    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    # metrics = metric(y_pred, y_pred, types=[])
    metric.confusion_matrix(y_true, y_pred)