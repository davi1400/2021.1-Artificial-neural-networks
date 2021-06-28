import sys
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
from mlfwk.readWrite import load_mock
from mlfwk.utils import split_random, get_project_root, normalization, out_of_c_to_label
from mlfwk.models import MultiLayerPerceptron
from mlfwk.readWrite import load_base
from mlfwk.visualization import generate_space, coloring

if __name__ == '__main__':
    print("run iris")
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
    base = load_base(path='iris.data', type='csv')

    # normalizar a base
    base[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] = normalization(
        base[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], type='min-max')

    N, M = base.shape
    C = len(base['Species'].unique())

    y_out_of_c = pd.get_dummies(base['Species'])

    base = base.drop(['Species'], axis=1)
    base = concatenate([base[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], y_out_of_c], axis=1)

    for realization in range(20):
        train, test = split_random(base, train_percentage=.8)
        train, train_val = split_random(train, train_percentage=.8)

        x_train = train[:, :4]
        y_train = train[:, 4:]

        x_train_val = train_val[:, :4]
        y_train_val = train_val[:, 4:]

        x_test = test[:, :4]
        y_test = test[:, 4:]

        validation_alphas = [0.15]
        hidden = 3 * np.arange(1, 5)
        simple_net = MultiLayerPerceptron(M, C, epochs=10000)
        simple_net.fit(x_train, y_train, x_train_val=x_train_val, y_train_val=y_train_val, alphas=validation_alphas, hidden=hidden)

        y_out_simple_net = simple_net.predict(x_test, bias=True)

        y_test = simple_net.predicao(y_test)


        metrics_calculator = metric(y_test, y_out_simple_net, types=['ACCURACY', 'precision', 'recall', 'f1_score'])
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
    #     df_cm = DataFrame(final_result['best_cf'][i], index=[i for i in ['']],
    #                          columns=[i for i in ['']])
    #     sn.heatmap(df_cm, annot=True)
    #
    #     path = get_project_root() + '/run/TR-03/IRIS/results/'
    #     plt.savefig(path + "mat_confsuison_iris.jpg")
    #     plt.show()

    print(pd.DataFrame(final_result))
    # del final_result['best_cf']
    pd.DataFrame(final_result).to_csv(get_project_root() + '/run/TR-03/IRIS/results/' + 'result_simple_net.csv')
