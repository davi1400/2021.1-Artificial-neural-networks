import numpy as np
from numpy import zeros, where, ndarray, array
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

    def calculate(self):
        result = {}
        for metric in self.types:
            result.update({metric: self.one_by_one(metric, self.y_real, self.y_pred)})

        return result

    def accuracy(self, y_true, y_pred):
        cf = self.confusion_matrix(y_true, y_pred)
        return (cf[0][0]+cf[1][1])/sum(sum(cf))

    def precision(self, y_true, y_pred):
        cf = self.confusion_matrix(y_true, y_pred)
        return cf[0][0]/(cf[0][0]+cf[1][0])

    def recall(self, y_true, y_pred):
        cf = self.confusion_matrix(y_true, y_pred)
        return cf[0][0]/(cf[0][0]+cf[0][1])

    def f1(self, y_true, y_pred):
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)

        return (2*precision*recall)/(precision+recall)

    def one_by_one(self, type, y_true, y_pred):

        # o AUC e MCC estão sendo calculados com o sklearn, TODO -> criar função para calculalos na mão
        func_possibles = {
            'ACCURACY': self.accuracy,
            'AUC': roc_auc_score,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1,
            'MCC': matthews_corrcoef
        }

        return func_possibles[type](y_true, y_pred)

    @staticmethod
    def confusion_matrix(y_true, y_pred):
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