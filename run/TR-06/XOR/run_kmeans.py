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
from mlfwk.models import kmeans
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
    base = base.to_numpy()

    # --------------------------------------------------------------------------------------

    for realization in range(1):
        train, test = split_random(base, train_percentage=.8)

        x_train = train[:, :2]
        y_train = train[:, 2:]

        x_test = test[:, :2]
        y_test = test[:, 2:]

        clf = kmeans(k=4, epsilon=1e-4, max_iter=1000)
        clf.fit(x_train, y_train, validation=False)

        y_out = clf.predict(x_test)


        # metrics_calculator = metric(y_test, y_out, types=['ACCURACY', 'precision', 'recall', 'f1_score'])
        # metric_results = metrics_calculator.calculate(average='macro')
        # print(metric_results)



    # ------------------------ PLOT -------------------------------------------------
    xx, yy = generate_space(x)
    space = c_[xx.ravel(), yy.ravel()]

    point = {
        0: 'bo',
        1: 'go',
        2: 'ko',
        3: 'ro',
    }
    marker = {
        0: '^',
        1: 'o',
        2: '*',
        3: '1'
    }

    # O clasificador da vigesima realização
    plot_dict = {
        'xx': xx,
        'yy': yy,
        'Z': clf.predict(space),
        'classes': {}
    }

    # utilizando o x_test e o y_test da ultima realização
    for c in [0, 1, 2, 3]:
        plot_dict['classes'].update({
            c: {
                'X': x[where(y == c)[0]],
                'point': point[c],
                'marker': marker[c]
            }
        })

    # #FFAAAA red
    # #AAAAFF blue
    coloring(plot_dict, ListedColormap(['#87CEFA', '#228B22', '#EAADEA', '#FF0000']), xlabel='x1', ylabel='x2',
             title='mapa de cores com mlp', xlim=[-0.1, 1.1], ylim=[-0.1, 1.1],
             path=get_project_root() + '/run/TR-05/XOR/results/' + 'color_map_xor_mlp_net.jpg',
             save=True)
    # print('dataset shape %s' % Counter(base[:, 2]))

    # print(pd.DataFrame(final_result))
    # # del final_result['best_cf']
    # pd.DataFrame(final_result).to_csv(get_project_root() + '/run/TR-05/XOR/results/' + 'result_mlp_net.csv')
    #
