import sys
from pathlib import Path
print(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import seaborn as sn
from mlfwk.readWrite import load_mock
import matplotlib.pyplot as plt
from numpy import where, mean, std, c_, array, linspace
from pandas.core.frame import DataFrame
from mlfwk.utils import split_random, get_project_root
from collections import Counter
from mlfwk.models import perceptron
from mlfwk.metrics import metric
from mlfwk.visualization import generate_space, coloring
from matplotlib.colors import ListedColormap

if __name__ == '__main__':
    final_result = {
        'ACCURACY': [],
        'std ACCURACY': [],
        'AUC': [],
        'std AUC': [],
        'MCC': [],
        'std MCC': [],
        'f1_score': [],
        'std f1_score': [],
        'precision': [],
        'std precision': [],
        'recall': [],
        'std recall': [],
        'best_cf': [],
        'ErrosxEpocohs': [],
        'alphas': []
    }

    results = {
        'realization': [],
        'ACCURACY': [],
        'AUC': [],
        'MCC': [],
        'f1_score': [],
        'precision': [],
        'recall': [],
        'cf': [],
        'erros': [],
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

    for realization in range(20):
        train, test = split_random(base, train_percentage=.8)
        train, train_val = split_random(train, train_percentage=0.7)

        x_train = train[:, :2]
        y_train = train[:, 2]

        x_train_val = train_val[:, :2]
        y_train_val = train_val[:, 2]

        x_test = test[:, :2]
        y_test = test[:, 2]

        classifier_perceptron = perceptron(epochs=500, learning_rate=0.01)

        classifier_perceptron.fit(x_train, y_train, x_train_val,
                                  y_train_val, alphas=validation_alphas)

        y_out_perceptron = classifier_perceptron.predict(x_test)

        metrics_calculator = metric(list(y_test), y_out_perceptron, types=['ACCURACY', 'AUC', 'precision',
                                                                    'recall', 'f1_score', 'MCC'])
        metric_results = metrics_calculator.calculate()

        if metric_results['ACCURACY'] == 1.0:
            pass

        results['cf'].append((metric_results['ACCURACY'], metrics_calculator.confusion_matrix(list(y_test),
                                                                                              y_out_perceptron)))
        results['erros'].append(classifier_perceptron.errors_per_epoch)
        results['alphas'].append(classifier_perceptron.learning_rate)
        results['realization'].append(realization)
        for type in ['ACCURACY', 'AUC', 'precision', 'recall', 'f1_score', 'MCC']:
            results[type].append(metric_results[type])



    results['cf'].sort(key=lambda x: x[0], reverse=True)
    final_result['best_cf'].append(results['cf'][0][1])
    final_result['ErrosxEpocohs'].append(results['erros'][0])
    final_result['alphas'].append(mean(results['alphas']))
    for type in ['ACCURACY', 'AUC', 'precision', 'recall', 'f1_score', 'MCC']:
        final_result[type].append(mean(results[type]))
        final_result['std ' + type].append(std(results[type]))



    for i in range(len(final_result['best_cf'])):
        plt.figure(figsize=(10, 7))

        df_cm = DataFrame(final_result['best_cf'][i], index=[i for i in "01"],
                             columns=[i for i in "01"])
        sn.heatmap(df_cm, annot=True)

        path = get_project_root() + '/run/TR-01/IRIS/results/'
        plt.savefig(path + "mat_confsuison_and.jpg")
        plt.show()

    for i in range(len(final_result['ErrosxEpocohs'])):
        plt.plot(list(range(len(final_result['ErrosxEpocohs'][i]))), final_result['ErrosxEpocohs'][i],
                 label='Error')
        plt.xlabel('Epocas')
        plt.ylabel('Error')
        plt.legend(loc='upper right')

        path = get_project_root() + '/run/TR-01/IRIS/results/'
        plt.savefig(path + "EpochsXError_and.jpg")
        plt.show()

    xx, yy = generate_space(x_test)
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
        'Z': array(classifier_perceptron.predict(space, [0, 1])),
        'classes': {}
    }

    # utilizando o x_test e o y_test da ultima realização
    for c in [0, 1]:
        plot_dict['classes'].update({
            c: {
              'X':  x_test[where(y_test == c)[0]],
              'point': point[c],
              'marker': marker[c]
            }
        })

    # #FFAAAA red
    # #AAAAFF blue
    coloring(plot_dict, ListedColormap(['#FFAAAA', '#AAAAFF']), xlabel='x1', ylabel='x2', title='mapa de cores com perceptron',
             path=get_project_root() + '/run/TR-01/ARTIFICIAL/results/' + 'color_map_and_percptron.jpg', save=True)
    print('dataset shape %s' % Counter(base[:, 2]))

    del final_result['best_cf']
    del final_result['ErrosxEpocohs']

    DataFrame(final_result).to_csv(get_project_root() + '/run/TR-01/ARTIFICIAL/results/' + 'result_percptron.csv')