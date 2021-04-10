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

        # if metric == 'ACCURACY':
        #     accuracy_score(self.y_real, self.y_pred)
        # if metric == 'AUC':
        #     roc_auc_score(self.y_real, self.y_pred)
        # if metric == 'precision':
        #     precision_score(self.y_real, self.y_pred)
        # if metric == 'recall':
        #     recall_score(self.y_real, self.y_pred)
        # if metric == 'f1_score':
        #     f1_score(self.y_real, self.y_pred)
        # if metric == 'MCC':
        #     matthews_corrcoef(self.y_real, self.y_pred)

    @staticmethod
    def one_by_one(type, y_true, y_pred):
        func_possibles = {
            'ACCURACY': accuracy_score,
            'AUC': roc_auc_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1_score': f1_score,
            'MCC': matthews_corrcoef
        }

        return func_possibles[type](y_true, y_pred)


if __name__ == '__main__':
    from numpy import ones
    y_pred = ones((2, 1))
    # metrics = metric(y_pred, y_pred, types=[])
    metric.one_by_one('ACCURACY', y_pred, y_pred)