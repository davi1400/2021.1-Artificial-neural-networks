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
from mlfwk.models import RadialBasisFunction
from mlfwk.visualization import generate_space, coloring

if __name__ == '__main__':
    print("run artificial XOR")
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

    base = pd.DataFrame(load_mock(type='LOGICAL_XOR'), columns=['x1', 'x2', 'y'])
    base[['x1', 'x2']] = normalization(base[['x1', 'x2']], type='min-max')

    x = array(base[['x1', 'x2']])
    y = array(base[['y']])

    classe0 = x[np.where(y == 0)[0]]
    classe1 = x[np.where(y == 1)[0]]

    plt.plot(classe0[:, 0], classe0[:, 1], 'b^')
    plt.plot(classe1[:, 0], classe1[:, 1], 'go')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.savefig(get_project_root() + '/run/TR-05/XOR/results/' + 'dataset_xor_artificial.png')
    plt.show()

    # ----------------------- one - hot ---------------------------------------------------
    N, M = base.shape
    C = len(base['y'].unique())

    y_out_of_c = pd.get_dummies(base['y'])
    base = concatenate([base[['x1', 'x2']], y_out_of_c], axis=1)

    # --------------------------------------------------------------------------------------

    for realization in range(20):
        train, test = split_random(base, train_percentage=.8)
        train, train_val = split_random(train, train_percentage=.8)

        x_train = train[:, :2]
        y_train = train[:, 2:]

        x_train_val = train_val[:, :2]
        y_train_val = train_val[:, 2:]

        x_test = test[:, :2]
        y_test = test[:, 2:]

        validation_alphas = [0.15]
        hidden = 2 * np.arange(1, 4)
        simple_net = RadialBasisFunction()
        simple_net.fit(x_train, y_train, x_train_val=x_train_val, y_train_val=y_train_val, alphas=validation_alphas,
                       hidden=hidden, validation=False)

        y_out = simple_net.predict(x_test, bias=True)

        y_test = simple_net.predicao(y_test)

        metrics_calculator = metric(y_test, y_out, types=['ACCURACY', 'precision', 'recall', 'f1_score'])
        metric_results = metrics_calculator.calculate(average='macro')
        print(metric_results)

        results['cf'].append(
            (metric_results['ACCURACY'], metrics_calculator.confusion_matrix(y_test, y_out, labels=[0, 1])))
        results['alphas'].append(simple_net.learning_rate)
        results['realization'].append(realization)
        for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
            results[type].append(metric_results[type])

    results['cf'].sort(key=lambda x: x[0], reverse=True)
    final_result['best_cf'].append(results['cf'][0][1])
    final_result['alphas'].append(mean(results['alphas']))
    for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
        final_result[type].append(mean(results[type]))
        final_result['std ' + type].append(std(results[type]))

    # ------------------------ PLOT -------------------------------------------------

    for i in range(len(final_result['best_cf'])):
        plt.figure(figsize=(10, 7))

        df_cm = DataFrame(final_result['best_cf'][i], index=[i for i in "01"],
                          columns=[i for i in "01"])
        sn.heatmap(df_cm, annot=True)

        path = get_project_root() + '/run/TR-05/XOR/results/'
        plt.savefig(path + "mat_confsuison_triangle.jpg")
        plt.show()

    xx, yy = generate_space(x)
    space = c_[xx.ravel(), yy.ravel()]

    point = {
        0: 'bo',
        1: 'go'
    }
    marker = {
        0: '^',
        1: 'o'
    }

    # O clasificador da vigesima realização
    plot_dict = {
        'xx': xx,
        'yy': yy,
        'Z': simple_net.predict(space, bias=True),
        'classes': {}
    }

    # utilizando o x_test e o y_test da ultima realização
    for c in [0, 1]:
        plot_dict['classes'].update({
            c: {
                'X': x[where(y == c)[0]],
                'point': point[c],
                'marker': marker[c]
            }
        })

    # #FFAAAA red
    # #AAAAFF blue
    coloring(plot_dict, ListedColormap(['#87CEFA', '#228B22']), xlabel='x1', ylabel='x2',
             title='mapa de cores com mlp', xlim=[-0.1, 1.1], ylim=[-0.1, 1.1],
             path=get_project_root() + '/run/TR-05/XOR/results/' + 'color_map_xor_mlp_net.jpg',
             save=True)
    # print('dataset shape %s' % Counter(base[:, 2]))

    print(pd.DataFrame(final_result))
    # del final_result['best_cf']
    pd.DataFrame(final_result).to_csv(get_project_root() + '/run/TR-05/XOR/results/' + 'result_mlp_net.csv')

