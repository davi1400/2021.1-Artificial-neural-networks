import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
print(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from pandas import DataFrame
from matplotlib.colors import ListedColormap

from numpy import where, append, ones, array, zeros, mean, argmax, linspace, concatenate, c_, std
from mlfwk.metrics import metric
from mlfwk.readWrite import load_base
from mlfwk.utils import split_random, get_project_root, normalization, out_of_c_to_label
from mlfwk.models import hyperbolic_perceptron_network
from mlfwk.visualization import generate_space, coloring


if __name__ == '__main__':

    versus = ['S_vs_OT', 'Vc_vs_OT', 'Vg_vs_OT']
    final_result = {
        'versus': [],
        'ACCURACY': [],
        'std ACCURACY': [],
        'AUC': [],
        'std AUC': [],
        # 'MCC': [],
        # 'std MCC': [],
        'f1_score': [],
        'std f1_score': [],
        'precision': [],
        'std precision': [],
        'recall': [],
        'std recall': [],
        'alphas': [],
        'best_cf': [],
        'ErrosxEpocohs': []
    }

    for one_versus_others in versus:

        # --------------------- load base ------------------------------------------------------------ #

        # carregar a base
        iris_base = load_base(path='iris.data', type='csv')

        # normalizar a base
        iris_base[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] = normalization(
            iris_base[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], type='min-max')

        setosa_ind = where(iris_base['Species'] == 'Iris-setosa')[0]
        versicolor_ind = where(iris_base['Species'] == 'Iris-versicolor')[0]
        virginica_ind = where(iris_base['Species'] == 'Iris-virginica')[0]

        if one_versus_others == 'S_vs_OT':
            # setosa versus others
            print('setosa versus others')
            iris_base['Species'].iloc[setosa_ind] = 1
            iris_base['Species'].iloc[versicolor_ind] = -1
            iris_base['Species'].iloc[virginica_ind] = -1

        elif one_versus_others == 'Vc_vs_OT':
            # versicolor versus others
            print('versicolor versus others')
            iris_base['Species'].iloc[setosa_ind] = -1
            iris_base['Species'].iloc[versicolor_ind] = 1
            iris_base['Species'].iloc[virginica_ind] = -1

        elif one_versus_others == 'Vg_vs_OT':
            # virginica versus others
            print('virginica versus others')
            iris_base['Species'].iloc[setosa_ind] = -1
            iris_base['Species'].iloc[versicolor_ind] = -1
            iris_base['Species'].iloc[virginica_ind] = 1
        # ----------------------------------------------------------------------------------------------- #

        accuracys = []
        results = {
            'versus': [],
            'realization': [],
            'ACCURACY': [],
            'AUC': [],
            # 'MCC': [],
            'f1_score': [],
            'precision': [],
            'recall': [],
            'alphas': [],
            'cf': [],
            'erros': []
        }
        validation_alphas = linspace(0.015, 0.01, 20)

        for realization in range(1):
            train, test = split_random(iris_base, train_percentage=0.8)
            train, train_val = split_random(train, train_percentage=0.8)

            x_train = train.drop(['Species'], axis=1)
            y_train = train['Species']

            x_train_val = train_val.drop(['Species'], axis=1)
            y_train_val = train_val['Species']

            x_test = test.drop(['Species'], axis=1)
            y_test = test['Species']

            classifier_perceptron = hyperbolic_perceptron_network(epochs=10000, learning_rate=0.01)

            classifier_perceptron.fit(x_train.to_numpy(), y_train.to_numpy(), x_train_val.to_numpy(),
                                      y_train_val.to_numpy(), alphas=validation_alphas)


            y_out_perceptron = classifier_perceptron.predict(x_test.to_numpy())

            # y_out_perceptron[np.where(y_out_perceptron == -1)] = 0

            metrics_calculator = metric(list(y_test), y_out_perceptron, types=['ACCURACY', 'AUC', 'precision',
                                                                  'recall', 'f1_score'])

            metric_results = metrics_calculator.calculate(average='macro')
            results['cf'].append((metric_results['ACCURACY'], metrics_calculator.confusion_matrix(list(y_test),
                                                                                                y_out_perceptron,
                                                                                                  [-1, 1])))

            results['erros'].append((metric_results['ACCURACY'], classifier_perceptron.errors_per_epoch))
            results['alphas'].append(classifier_perceptron.learning_rate)
            results['realization'].append(realization)
            for type in ['ACCURACY', 'AUC', 'precision', 'recall', 'f1_score']:
                results[type].append(metric_results[type])



        results['cf'].sort(key=lambda x: x[0], reverse=True)
        results['erros'].sort(key=lambda x: x[0], reverse=True)

        final_result['best_cf'].append(results['cf'][0][1])
        final_result['ErrosxEpocohs'].append(results['erros'][0][1])
        final_result['versus'].append(one_versus_others)
        final_result['alphas'].append(mean(results['alphas']))
        for type in ['ACCURACY', 'AUC', 'precision', 'recall', 'f1_score']:
            final_result[type].append(mean(results[type]))
            final_result['std ' + type].append(std(results[type]))

    for i in range(len(final_result['best_cf'])):
        plt.figure(figsize=(10, 7))

        df_cm = DataFrame(final_result['best_cf'][i], index=[i for i in "01"],
                             columns=[i for i in "01"])
        sn.heatmap(df_cm, annot=True)

        path = get_project_root() + '/run/TR-035/IRIS/results/'
        plt.savefig(path + 'mat_confsuison_hyper' + final_result['versus'][i] + ".jpg")
        plt.show()

    # for i in range(len(final_result['ErrosxEpocohs'])):
    #     plt.plot(list(range(len(final_result['ErrosxEpocohs'][i]))), final_result['ErrosxEpocohs'][i],
    #              label='Error')
    #     plt.xlabel('Epocas')
    #     plt.ylabel('Error')
    #     plt.legend(loc='upper right')
    #
    #     path = get_project_root() + '/run/TR-01/IRIS/results/'
    #     plt.savefig(path + 'EpochsXError ' + final_result['versus'][i] + ".jpg")
    #     plt.show()
    #
    #

    del final_result['best_cf']
    del final_result['ErrosxEpocohs']
    DataFrame(final_result).to_csv(get_project_root() + '/run/TR-035/IRIS/results/' + 'result_perceptron.csv')
    print(DataFrame(final_result))
