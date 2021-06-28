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
from mlfwk.models import sigmoid_perceptron_network
from mlfwk.visualization import generate_space, coloring


if __name__ == '__main__':

    # carregar a base
    base = load_base(path='iris.data', type='csv')

    # normalizar a base
    base[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] = normalization(
        base[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], type='min-max')

    features_combinations = [
        ['SepalLengthCm', 'SepalWidthCm', 'Species'],
        ['SepalLengthCm', 'PetalLengthCm', 'Species'],
        ['SepalWidthCm', 'PetalLengthCm', 'Species'],
        ['PetalLengthCm', 'PetalWidthCm', 'Species']
    ]

    versus = ['S_vs_OT', 'Vc_vs_OT', 'Vg_vs_OT']

    for combination in features_combinations:
        for one_versus_others in versus:

            iris_base = base.copy()
            iris_base = iris_base[combination]

            setosa_ind = where(iris_base['Species'] == 'Iris-setosa')[0]
            versicolor_ind = where(iris_base['Species'] == 'Iris-versicolor')[0]
            virginica_ind = where(iris_base['Species'] == 'Iris-virginica')[0]

            if one_versus_others == 'S_vs_OT':
                # setosa versus others
                print('setosa versus others')
                iris_base['Species'].iloc[setosa_ind] = 1
                iris_base['Species'].iloc[versicolor_ind] = 0
                iris_base['Species'].iloc[virginica_ind] = 0

            elif one_versus_others == 'Vc_vs_OT':
                # versicolor versus others
                print('versicolor versus others')
                iris_base['Species'].iloc[setosa_ind] = 0
                iris_base['Species'].iloc[versicolor_ind] = 1
                iris_base['Species'].iloc[virginica_ind] = 0

            elif one_versus_others == 'Vg_vs_OT':
                # virginica versus others
                print('virginica versus others')
                iris_base['Species'].iloc[setosa_ind] = 0
                iris_base['Species'].iloc[versicolor_ind] = 0
                iris_base['Species'].iloc[virginica_ind] = 1


            # ----------------------- train to plot, just one realization ---------------------------
            validation_alphas = linspace(0.015, 0.01, 20)

            train, test = split_random(iris_base, train_percentage=0.8)
            train, train_val = split_random(train, train_percentage=0.7)

            x_train = train.drop(['Species'], axis=1)
            y_train = train['Species']

            x_train_val = train_val.drop(['Species'], axis=1)
            y_train_val = train_val['Species']

            x_test = test.drop(['Species'], axis=1)
            y_test = test['Species']

            classifier_perceptron = sigmoid_perceptron_network(epochs=10000, learning_rate=0.15)

            classifier_perceptron.fit(x_train.to_numpy(), y_train.to_numpy(),
                                      x_train_val.to_numpy(), y_train_val.to_numpy(),
                                      alphas=validation_alphas)

            x = array(iris_base[combination[:2]])
            y = array(iris_base[combination[2]])

            xx, yy = generate_space(x)
            space = c_[xx.ravel(), yy.ravel()]

            point = {
                0: 'bo',
                1: 'go',
            }
            marker = {
                0: '^',
                1: 'o',
            }

            # O clasificador da vigesima realização
            plot_dict = {
                'xx': xx,
                'yy': yy,
                'Z': classifier_perceptron.predict(space),
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

            path = get_project_root() + '/run/TR-035/IRIS/results/' + 'color_map_' + str(combination) + str(one_versus_others) + '.jpg'
            coloring(plot_dict, ListedColormap(['#87CEFA', '#228B22']), xlabel=combination[0],
                     ylabel=combination[1], title='mapa de cores com Rede Perceptron' + str(one_versus_others),
                     xlim=[-0.1, 1.1], ylim=[-0.1, 1.1],
                     path=path,
                     save=True)
