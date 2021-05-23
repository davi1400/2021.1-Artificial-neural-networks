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
from mlfwk.visualization import generate_space, coloring

if __name__ == '__main__':
    print("run artificial")
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

    # sem normalização
    base = load_mock(type='LOGICAL_AND')
    validation_alphas = linspace(0.015, 0.1, 20)

    pos = base[:, :2][where(base[:, 2] == 1)[0]]
    neg = base[:, :2][where(base[:, 2] == 0)[0]]
    plt.plot(pos[:, 0], pos[:, 1], 'bo')
    plt.plot(neg[:, 0], neg[:, 1], 'ro')
    plt.show()


    for realization in range(1):
        train, test = split_random(base, train_percentage=.8)
        train, train_val = split_random(train, train_percentage=.8)

        x_train = train[:, :2]
        y_train = train[:, 2:]

        x_train_val = train_val[:, :2]
        y_train_val = train_val[:, 2:]

        x_test = test[:, :2]
        y_test = test[:, 2:]

        validation_alphas = linspace(0.015, 0.1, 20)
        sigmoid_net = sigmoid_perceptron_network(epochs=1000, number_of_neurons=1, learning_rate=0.1, activation_function='sigmoid logistic')
        sigmoid_net.fit(x_train, y_train, x_train_val, y_train_val, alphas=validation_alphas, validation=False)

        y_out = sigmoid_net.predict(x_test)

        metrics_calculator = metric(y_test, y_out, types=['ACCURACY', 'precision', 'recall', 'f1_score'])
        metric_results = metrics_calculator.calculate(average='macro')
        print(metric_results)

        results['cf'].append((metric_results['ACCURACY'], metrics_calculator.confusion_matrix(y_test, y_out, labels=[0, 1])))
        results['alphas'].append(sigmoid_net.learning_rate)
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

        path = get_project_root() + '/run/TR-035/ARTIFICIAL/results/'
        plt.savefig(path + "mat_confsuison_triangle.jpg")
        plt.show()


    x = base[:, :2]
    y = base[:, 2]
    xx, yy = generate_space(x)
    space = c_[xx.ravel(), yy.ravel()]

    point = {
        0: 'ro',
        1: 'bo'
    }
    marker = {
        0: 's',
        1: 'D'
    }

    # O clasificador da vigesima realização
    plot_dict = {
        'xx': xx,
        'yy': yy,
        'Z': array(sigmoid_net.predict(space, [0, 1])),
        'classes': {}
    }

    # utilizando o x_test e o y_test da ultima realização
    for c in [0, 1]:
        plot_dict['classes'].update({
            c: {
                'X': x_test[where(y_test == c)[0]],
                'point': point[c],
                'marker': marker[c]
            }
        })


    # #FFAAAA red
    # #AAAAFF blue
    coloring(plot_dict, ListedColormap(['#FFAAAA', '#AAAAFF']), xlabel='x1', ylabel='x2',
             title='mapa de cores com Perceptron sigmoid',
             path=get_project_root() + '/run/TR-035/ARTIFICIAL/results/' + 'color_map_triangle_sigmoid_net.jpg', save=True)
    # print('dataset shape %s' % Counter(base[:, 2]))

    print(pd.DataFrame(final_result))
    # del final_result['best_cf']
    pd.DataFrame(final_result).to_csv(get_project_root() + '/run/TR-035/ARTIFICIAL/results/' + 'result_sigmoid_net.csv')
