import sys
from pathlib import Path
print(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from pandas import DataFrame
from matplotlib.colors import ListedColormap

from numpy import where, append, ones, array, zeros, mean, argmax, linspace, concatenate, c_, std
from mlfwk.metrics import metric
from mlfwk.utils import split_random, get_project_root, normalization, out_of_c_to_label
from mlfwk.models import RadialBasisFunction
from mlfwk.readWrite import load_base
from mlfwk.visualization import generate_space, coloring


def predicao(Y):
    y = np.zeros((Y.shape[0], 1))
    for j in range(Y.shape[0]):
        i = np.where(Y[j, :] == Y[j, :].max())[0][0]
        y[j] = i
    return y


if __name__ == '__main__':
    print("run coluna 2 classes")
    final_result = {
        'ACCURACY': [],
        'std ACCURACY': [],
        # 'MCC': [],
        # 'std MCC': [],
        'f1_score': [],
        'std f1_score': [],
        'precision': [],
        'std precision': [],
        'recall': [],
        'std recall': [],
        'best_cf': [],
        'alphas': []
    }

    results = {
        'realization': [],
        'ACCURACY': [],
        # 'MCC': [],
        'f1_score': [],
        'precision': [],
        'recall': [],
        'cf': [],
        'alphas': []
    }

    # carregar a base
    base = load_base(path='column_2C_weka.arff', type='arff')


    # features
    features = ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis']

    print(base.info())

    # ----------------------------- Clean the data ----------------------------------------------------------------

    # -------------------------- Normalization ------------------------------------------------------------------

    # normalizar a base
    base[features] = normalization(base[features], type='min-max')


    # ------------------------------------------------------------------------------------------------------------

    N, M = base.shape
    C = len(base['class'].unique())

    y_out_of_c = pd.get_dummies(base['class'])

    base = base.drop(['class'], axis=1)
    base = concatenate([base[features], y_out_of_c], axis=1)

    for realization in range(20):
        train, test = split_random(base, train_percentage=.8)
        train, train_val = split_random(train, train_percentage=.8)

        x_train = train[:, :len(features)]
        y_train = train[:, len(features):]

        x_train_val = train_val[:, :len(features)]
        y_train_val = train_val[:, len(features):]

        x_test = test[:, :len(features)]
        y_test = test[:, len(features):]


        validation_alphas = [0.3, 0.5, 1.0, 2.0]
        hidden = [10, 15, 20]
        simple_net = RadialBasisFunction(number_of_neurons=15, N_Classes=2, alpha=0.5, case='classification')
        simple_net.fit(x_train, y_train, x_train_val=x_train_val, y_train_val=y_train_val, alphas=validation_alphas,
                       hidden=hidden)

        y_out = simple_net.predict(x_test, bias=True)
        y_test = simple_net.predicao(y_test)

        metrics_calculator = metric(y_test, y_out, types=['ACCURACY', 'precision', 'recall', 'f1_score'])
        metric_results = metrics_calculator.calculate(average='macro')
        print(metric_results)

        results['cf'].append((metric_results['ACCURACY'],
                              metrics_calculator.confusion_matrix(list(y_test), y_out, labels=list(range(C))),
                              simple_net.alpha,
                              simple_net.number_of_neurons
                              ))

        results['alphas'].append(simple_net.alpha)
        results['realization'].append(realization)
        for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
            results[type].append(metric_results[type])

    results['cf'].sort(key=lambda x: x[0], reverse=True)
    final_result['best_cf'].append(results['cf'][0][1])
    best_alpha = results['cf'][0][2]
    best_number_centers = results['cf'][0][3]


    final_result['alphas'].append(mean(results['alphas']))
    for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
        final_result[type].append(mean(results[type]))
        final_result['std ' + type].append(std(results[type]))


    # ------------------------ PLOT -------------------------------------------------

    for i in range(len(final_result['best_cf'])):
        plt.figure(figsize=(10, 7))

        df_cm = DataFrame(final_result['best_cf'][i], index=[i for i in range(C)],
                             columns=[i for i in range(C)])
        sn.heatmap(df_cm, annot=True)
        plt.title(
            'Matriz de connfusão dermatologia com raio de abertura: ' + str(best_alpha) + ' e número de neurônios: ' + str(
                best_number_centers))
        plt.xlabel('Valor Esperado')
        plt.ylabel('Valor Encontrado')

        path = get_project_root() + '/run/TR-06/COLUNA_2C/results/'
        plt.savefig(path + "mat_confsuison_rbf.jpg")
        plt.show()

    print(pd.DataFrame(final_result))
    del final_result['best_cf']
    pd.DataFrame(final_result).to_csv(get_project_root() + '/run/TR-06/COLUNA_2C/results/' + 'result_rbf_net.csv')
