import sys
from pathlib import Path
print(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


from mlfwk.readWrite import load_mock
import matplotlib.pyplot as plt
from numpy import where, mean, std, c_, array
from pandas.core.frame import DataFrame
from mlfwk.utils import split_random, get_project_root
from collections import Counter
from mlfwk.models import dmc
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
        'std recall': []
    }

    results = {
        'realization': [],
        'ACCURACY': [],
        'AUC': [],
        'MCC': [],
        'f1_score': [],
        'precision': [],
        'recall': []
    }

    # sem normalização
    base = load_mock(type='LOGICAL_AND')

    pos = base[:, :2][where(base[:, 2] == 1)[0]]
    neg = base[:, :2][where(base[:, 2] == 0)[0]]
    plt.plot(pos[:, 0], pos[:, 1], 'bo')
    plt.plot(neg[:, 0], neg[:, 1], 'ro')
    plt.show()

    for realization in range(20):
        train, test = split_random(base, train_percentage=.8)

        x_train = train[:, :2]
        y_train = train[:, 2]

        x_test = test[:, :2]
        y_test = test[:, 2]

        classifier_dmc = dmc(x_train, y_train)
        y_out_dmc = classifier_dmc.predict(x_test, [0, 1])

        metrics_calculator = metric(list(y_test), y_out_dmc, types=['ACCURACY', 'AUC', 'precision',
                                                                    'recall', 'f1_score', 'MCC'])
        metric_results = metrics_calculator.calculate()

        results['realization'].append(realization)
        for type in ['ACCURACY', 'AUC', 'precision', 'recall', 'f1_score', 'MCC']:
            results[type].append(metric_results[type])

    for type in ['ACCURACY', 'AUC', 'precision', 'recall', 'f1_score', 'MCC']:
        final_result[type].append(mean(results[type]))
        final_result['std ' + type].append(std(results[type]))

    DataFrame(final_result).to_csv(get_project_root() + '/run/TR-00/ARTIFICIAL/results/' + 'result_dmc.csv')

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
        'Z': array(classifier_dmc.predict(space, [0, 1])),
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
    coloring(plot_dict, ListedColormap(['#FFAAAA', '#AAAAFF']), xlabel='x1', ylabel='x2', title='mapa de cores com dmc',
             path='color_map_and_dmc.jpg')
    print('dataset shape %s' % Counter(base[:, 2]))