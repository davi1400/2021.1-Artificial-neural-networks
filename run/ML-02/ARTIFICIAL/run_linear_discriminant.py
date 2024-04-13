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
from mlfwk.models import linearDiscriminant
from mlfwk.visualization import generate_space, coloring

if __name__ == '__main__':
    print("run artificial")
    final_result = {
        'ACCURACY': [],
        'std ACCURACY': [],
        'f1_score': [],
        'std f1_score': [],
        'precision': [],
        'std precision': [],
        'recall': [],
        'std recall': [],
        'best_cf': [],
    }

    results = {
        'realization': [],
        'ACCURACY': [],
        'f1_score': [],
        'precision': [],
        'recall': [],
        'cf': []
    }

    base = load_mock(type='TRIANGLE_CLASSES')
    # normalizar a base
    features = ['x1', 'x2']

    base[features] = normalization(base[features], type='min-max')
    classes = base['y'].unique()

    x = array(base[features])
    y = array(base[['y']])

    classe0 = x[np.where(y == 0)[0]]
    classe1 = x[np.where(y == 1)[0]]
    classe2 = x[np.where(y == 2)[0]]

    plt.plot(classe0[:, 0], classe0[:, 1], 'b^')
    plt.plot(classe1[:, 0], classe1[:, 1], 'go')
    plt.plot(classe2[:, 0], classe2[:, 1], 'm*')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.savefig(get_project_root() + '/run/ML-02/ARTIFICIAL/results/' + 'dataset_artificial.png')
    plt.show()

    for realization in range(20):
        train, test = split_random(base, train_percentage=.8)

        linear_clf = linearDiscriminant(classes, features, 'y')
        linear_clf.fit(train)
        y_out = linear_clf.predict(test)

        metrics_calculator = metric(list(test['y']), y_out, types=['ACCURACY', 'precision', 'recall', 'f1_score'])
        metric_results = metrics_calculator.calculate(average='macro')
        print(metric_results)

        results['cf'].append((metric_results['ACCURACY'], metrics_calculator.confusion_matrix(list(test['y']), y_out, labels=[0, 1, 2])))
        results['realization'].append(realization)
        for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
            results[type].append(metric_results[type])

    results['cf'].sort(key=lambda x: x[0], reverse=True)
    final_result['best_cf'].append(results['cf'][0][1])
    for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
        final_result[type].append(mean(results[type]))
        final_result['std ' + type].append(std(results[type]))


    # ------------------------ PLOT -------------------------------------------------

    for i in range(len(final_result['best_cf'])):
        plt.figure(figsize=(10, 7))

        df_cm = DataFrame(final_result['best_cf'][i], index=[i for i in "012"],
                             columns=[i for i in "012"])
        sn.heatmap(df_cm, annot=True)

        path = get_project_root() + '/run/ML-02/ARTIFICIAL/results/'
        plt.savefig(path + "mat_confsuison_triangle_LD.jpg")
        plt.show()



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
        'Z': array(linear_clf.predict(DataFrame(space, columns=features))),
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
    coloring(plot_dict, ListedColormap(['#87CEFA', '#228B22', "#FF00FF"]), xlabel='x1', ylabel='x2',
             title='mapa de cores com discriminante linear', xlim=[-0.1, 1.1], ylim=[-0.1, 1.1],
             path=get_project_root() + '/run/ML-02/ARTIFICIAL/results/' + 'color_map_triangle_LD.jpg', save=True)

    print(pd.DataFrame(final_result))
    pd.DataFrame(final_result).to_csv(get_project_root() + '/run/ML-02/ARTIFICIAL/results/' + 'result_LD.csv')
