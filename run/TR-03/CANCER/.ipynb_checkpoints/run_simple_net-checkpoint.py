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
from mlfwk.readWrite import load_mock
from mlfwk.utils import split_random, get_project_root, normalization, out_of_c_to_label
from mlfwk.models import simple_perceptron_network
from mlfwk.readWrite import load_base
from mlfwk.visualization import generate_space, coloring

if __name__ == '__main__':
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
        # 'best_cf': [],
        'alphas': []
    }

    results = {
        'realization': [],
        'ACCURACY': [],
        # 'MCC': [],
        'f1_score': [],
        'precision': [],
        'recall': [],
        # 'cf': [],
        'alphas': []
    }

    # carregar a base
    base = load_base(path='breast-cancer-wisconsin.data', type='csv')
    base = base.drop(['Sample code number'], axis=1)

    # features
    features = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
       'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']

    print(base.info())

    # ----------------------------- Clean the data ----------------------------------------------------------------

    # The values at the column Bare Nuclei are all strings so we have to transform to int each of them.
    for unique_value in base['Bare Nuclei']:
        if unique_value != '?':
            base['Bare Nuclei'][base['Bare Nuclei'] == unique_value] = int(unique_value)

    # ? -> mean of column
    base['Bare Nuclei'][base['Bare Nuclei'] == '?'] = int(np.mean(base['Bare Nuclei'][base['Bare Nuclei'] != '?']))

    # -------------------------- Normalization ------------------------------------------------------------------

    # normalizar a base
    base[features] = normalization(base[features], type='min-max')


    # ------------------------------------------------------------------------------------------------------------

    y_out_of_c = pd.get_dummies(base['Class'])

    base = base.drop(['Class'], axis=1)
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

        validation_alphas = linspace(0.01, 0.1, 20)
        simple_net = simple_perceptron_network(epochs=10000, number_of_neurons=2, learning_rate=0.01, activation_function='degree')
        simple_net.fit(x_train, y_train, x_train_val, y_train_val, alphas=validation_alphas)

        y_out_simple_net = simple_net.predict(x_test)
        y_out = out_of_c_to_label(y_out_simple_net)
        y_test = out_of_c_to_label(y_test)


        metrics_calculator = metric(y_test, y_out, types=['ACCURACY', 'precision', 'recall', 'f1_score'])
        metric_results = metrics_calculator.calculate(average='micro')
        print(metric_results)

        results['alphas'].append(simple_net.learning_rate)
        results['realization'].append(realization)
        for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
            results[type].append(metric_results[type])

    final_result['alphas'].append(mean(results['alphas']))
    for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
        final_result[type].append(mean(results[type]))
        final_result['std ' + type].append(std(results[type]))



    print(pd.DataFrame(final_result))
    # del final_result['best_cf']
    pd.DataFrame(final_result).to_csv(get_project_root() + '/run/TR-03/CANCER/results/' + 'result_simple_net.csv')
