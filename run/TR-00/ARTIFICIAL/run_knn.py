import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
print(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import seaborn as sn
from mlfwk.readWrite import load_mock
import matplotlib.pyplot as plt
from numpy import where, mean, std, c_, array
from pandas.core.frame import DataFrame
from mlfwk.utils import split_random, get_project_root
from collections import Counter
from mlfwk.models import knn
from mlfwk.metrics import metric
from mlfwk.visualization import generate_space, coloring
from matplotlib.colors import ListedColormap

if __name__ == '__main__':
    final_result = {
        'ACCURACY': [],
        'std ACCURACY': [],
        'f1_score': [],
        'std f1_score': [],
        'precision': [],
        'std precision': [],
        'recall': [],
        'std recall': [],
        'best_cf': []
    }

    results = {
        'realization': [],
        'ACCURACY': [],
        'AUC': [],
        'MCC': [],
        'f1_score': [],
        'precision': [],
        'recall': [],
        'cf': []
    }

    base = load_mock(type='LOGICAL_AND')

    pos = base[:, :2][where(base[:, 2] == 1)[0]]
    neg = base[:, :2][where(base[:, 2] == 0)[0]]
    plt.plot(pos[:, 0], pos[:, 1], 'bo')
    plt.plot(neg[:, 0], neg[:, 1], 'ro')
    plt.show()

    C = [0, 1]

    for realization in range(20):
        train, test = split_random(base, train_percentage=.8)

        x_train = train[:, :2]
        y_train = train[:, 2]

        x_test = test[:, :2]
        y_test = test[:, 2]

        classifier_knn = knn(x_train, y_train, k=3)
        y_out_knn = classifier_knn.predict(x_test)

        metrics_calculator = metric(list(y_test), y_out_knn, types=['ACCURACY', 'precision',
                                                                    'recall', 'f1_score'])
        metric_results = metrics_calculator.calculate(average='micro')

        results['cf'].append((metric_results['ACCURACY'],
                              metrics_calculator.confusion_matrix(list(y_test), y_out_knn, labels=[0, 1]),
                              classifier_knn
                              ))

        results['realization'].append(realization)
        for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
            results[type].append(metric_results[type])

    results['cf'].sort(key=lambda x: x[0], reverse=True)
    final_result['best_cf'].append(results['cf'][0][1])
    best_acc_clf = results['cf'][0][2]
    best_acc = results['cf'][0][0]

    for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
        final_result[type].append(mean(results[type]))
        final_result['std ' + type].append(std(results[type]))

    print(DataFrame(final_result))
    DataFrame(final_result).to_csv(get_project_root() + '/run/TR-00/ARTIFICIAL/results/' + 'result_knn.csv')

    for i in range(len(final_result['best_cf'])):
        plt.figure(figsize=(10, 7))

        df_cm = DataFrame(final_result['best_cf'][i], index=[i for i in range(2)],
                          columns=[i for i in range(2)])
        sn.heatmap(df_cm, annot=True)
        plt.title('Matriz de connfusão do KNN com acurácia de ' + str(best_acc*100) + "%")
        plt.xlabel('Valor Esperado')
        plt.ylabel('Valor Encontrado')

        path = get_project_root() + '/run/TR-00/ARTIFICIAL/results/'
        plt.savefig(path + "conf_result_knn.jpg")
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
        'Z': array(best_acc_clf.predict(space)),
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
    coloring(plot_dict, ListedColormap(['#FFAAAA', '#AAAAFF']), xlabel='x1', ylabel='x2', title='mapa de cores com knn',
             path=path + 'color_map_and_knn.jpg', save=True)
    print('dataset shape %s' % Counter(base[:, 2]))