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
from mlfwk.readWrite import load_mock
from mlfwk.utils import split_random, get_project_root, normalization, out_of_c_to_label
from mlfwk.models import  MultiLayerPerceptron
from mlfwk.readWrite import load_base
from mlfwk.visualization import generate_space, coloring

if __name__ == '__main__':
    print("run dermatologia")
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
    columns = []
    for i in range(34):
        columns.append('x'+str(i))

    columns.append('y')
    base = load_base(path='dermatology.data', column_names=columns, type='csv')



    # features
    features = columns[:len(columns)-1]
    print(base.info())

    # ----------------------------- Clean the data ----------------------------------------------------------------

    # The Age has values ?
    for unique_value in base['x33'].unique():
        if unique_value != '?':
            base['x33'][base['x33'] == unique_value] = int(unique_value)

    # ? -> mean of column
    base['x33'][base['x33'] == '?'] = int(np.mean(base['x33'][base['x33'] != '?']))

    # -------------------------- Normalization ------------------------------------------------------------------

    # normalizar a base
    base[features] = (normalization(base[features], type='min-max')).to_numpy(dtype=np.float)


    # ------------------------------------------------------------------------------------------------------------

    N, M = base.shape
    C = len(base['y'].unique())

    y_out_of_c = pd.get_dummies(base['y'])

    base = base.drop(['y'], axis=1)
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

        validation_alphas = [0.15, 0.05, 0.1, 0.1]
        hidden = 3 * np.arange(1, 6)
        simple_net = MultiLayerPerceptron(M, C, epochs=10000)
        simple_net.fit(x_train, y_train, x_train_val=x_train_val, y_train_val=y_train_val, alphas=validation_alphas,
                       hidden=hidden)

        y_out = simple_net.predict(x_test, bias=True)

        y_test = simple_net.predicao(y_test)


        metrics_calculator = metric(y_test, y_out, types=['ACCURACY', 'precision', 'recall', 'f1_score'])
        metric_results = metrics_calculator.calculate(average='macro')
        print(metric_results)

        results['alphas'].append(simple_net.learning_rate)
        results['realization'].append(realization)
        for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
            results[type].append(metric_results[type])

    final_result['alphas'].append(mean(results['alphas']))
    for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
        final_result[type].append(mean(results[type]))
        final_result['std ' + type].append(std(results[type]))


    # ------------------------ PLOT -------------------------------------------------

    # for i in range(len(final_result['best_cf'])):
    #     plt.figure(figsize=(10, 7))
    #
    #     df_cm = DataFrame(final_result['best_cf'][i], index=[i for i in "012"],
    #                          columns=[i for i in "012"])
    #     sn.heatmap(df_cm, annot=True)
    #
    #     path = get_project_root() + '/run/TR-03/ARTIFICIAL/results/'
    #     plt.savefig(path + "mat_confsuison_triangle.jpg")
    #     plt.show()


    print(pd.DataFrame(final_result))
    # del final_result['best_cf']
    pd.DataFrame(final_result).to_csv(get_project_root() + '/run/TR-05/DERMATOLOGIA/results/' + 'result_simple_net.csv')
