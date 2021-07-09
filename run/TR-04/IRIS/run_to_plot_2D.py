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
from mlfwk.models import sigmoid_perceptron_network
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
    base = base.drop(['PetalLengthCm', 'PetalWidthCm'], axis=1)

    # normalizar a base
    base[['SepalLengthCm', 'SepalWidthCm']] = normalization(
        base[['SepalLengthCm', 'SepalWidthCm']], type='min-max')


    y_out_of_c = pd.get_dummies(base['Species'])

    base = base.drop(['Species'], axis=1)
    base = concatenate([base[['SepalLengthCm', 'SepalWidthCm']], y_out_of_c], axis=1)


    train, test = split_random(base, train_percentage=.8)
    train, train_val = split_random(train, train_percentage=.8)

    x_train = train[:, :2]
    y_train = train[:, 2:]

    x_train_val = train_val[:, :2]
    y_train_val = train_val[:, 2:]

    x_test = test[:, :2]
    y_test = test[:, 2:]

    validation_alphas = linspace(0.01, 0.1, 20)
    simple_net = sigmoid_perceptron_network(epochs=1000, number_of_neurons=3, learning_rate=0.01)
    simple_net.fit(x_train, y_train, x_train_val, y_train_val, alphas=validation_alphas)

    y_out_simple_net = simple_net.predict(x_test)
    y_out = out_of_c_to_label(y_out_simple_net)
    y_test = out_of_c_to_label(y_test)

    metrics_calculator = metric(y_test, y_out, types=['ACCURACY', 'precision', 'recall', 'f1_score'])
    metric_results = metrics_calculator.calculate(average='macro')
    print(metric_results)

    x = x_test.copy()
    y = y_test.copy()

    xx, yy = generate_space(x)
    space = c_[xx.ravel(), yy.ravel()]

    point = {
        0: 'bo',
        1: 'go',
        2: 'mo'
    }
    marker = {
        0: '^',
        1: 'o',
        2: '*'
    }

    # O clasificador da vigesima realização
    plot_dict = {
        'xx': xx,
        'yy': yy,
        'Z': out_of_c_to_label(simple_net.predict(space)),
        'classes': {}
    }

    # utilizando o x_test e o y_test da ultima realização
    for c in [0, 1, 2]:
        plot_dict['classes'].update({
            c: {
                'X': x[where(y == c)[0]],
                'point': point[c],
                'marker': marker[c]
            }
        })

    # #FFAAAA red
    # #AAAAFF blue
    coloring(plot_dict, ListedColormap(['#87CEFA', '#228B22', "#FF00FF"]), xlabel='SepalLengthCm',
             ylabel='SepalWidthCm',
             title='mapa de cores com Rede Perceptron - ACC: ' + str(metric_results['ACCURACY'].round(2)), xlim=[-0.1, 1.1], ylim=[-0.1, 1.1],
             path=get_project_root() + '/run/TR-04/IRIS/results/' + 'color_map_sepal_test.jpg', save=True)
    # print('dataset shape %s' % Counter(base[:, 2]))

    # ------------------ All points -------------------------------------------------------------------

    x = array(base[:, :2])
    y = array(out_of_c_to_label(base[:, 2:]))


    xx, yy = generate_space(x)
    space = c_[xx.ravel(), yy.ravel()]

    point = {
        0: 'bo',
        1: 'go',
        2: 'mo'
    }
    marker = {
        0: '^',
        1: 'o',
        2: '*'
    }

    # O clasificador da vigesima realização
    plot_dict = {
        'xx': xx,
        'yy': yy,
        'Z': out_of_c_to_label(simple_net.predict(space)),
        'classes': {}
    }

    # utilizando o x_test e o y_test da ultima realização
    for c in [0, 1, 2]:
        plot_dict['classes'].update({
            c: {
                'X': x[where(y == c)[0]],
                'point': point[c],
                'marker': marker[c]
            }
        })

    # #FFAAAA red
    # #AAAAFF blue
    coloring(plot_dict, ListedColormap(['#87CEFA', '#228B22', "#FF00FF"]), xlabel='SepalLengthCm', ylabel='SepalWidthCm',
             title='mapa de cores com Rede Perceptron', xlim=[-0.1, 1.1], ylim=[-0.1, 1.1],
             path=get_project_root() + '/run/TR-04/IRIS/results/' + 'color_map_sepal_all_points.jpg', save=True)
    # print('dataset shape %s' % Counter(base[:, 2]))



