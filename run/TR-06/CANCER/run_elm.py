import sys
from pathlib import Path
print(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import warnings
warnings.filterwarnings("ignore")

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
from mlfwk.models import ExtremeLearningMachines
from mlfwk.readWrite import load_base
from mlfwk.visualization import generate_space, coloring

if __name__ == '__main__':
    print("run breast cancer")
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
        'best_cf': []
    }

    results = {
        'realization': [],
        'ACCURACY': [],
        # 'MCC': [],
        'f1_score': [],
        'precision': [],
        'recall': [],
        'cf': []
    }

    # carregar a base
    base = load_base(path='breast-cancer-wisconsin.data', type='csv')
    base = base.drop(['Sample code number'], axis=1)

    # features
    features = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
       'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']

    print(base.info())

    # ----------------------------- Clean the data ----------------------------------------------------------------

    # The values at the column Bare Nuclei are all strings so we have to transform to int each of them.
    for unique_value in base['Bare Nuclei']:
        if unique_value != '?':
            base['Bare Nuclei'][base['Bare Nuclei'] == unique_value] = int(unique_value)

    # ? -> mean of column
    base['Bare Nuclei'][base['Bare Nuclei'] == '?'] = int(np.mean(base['Bare Nuclei'][base['Bare Nuclei'] != '?']))

    # -------------------------- Normalization ------------------------------------------------------------------

    # normalizar a base
    base[features] = (normalization(base[features], type='min-max')).to_numpy(dtype=np.float)


    # ------------------------------------------------------------------------------------------------------------
    N, M = base.shape
    C = len(base['Class'].unique())

    y_out_of_c = pd.get_dummies(base['Class'])

    base = base.drop(['Class'], axis=1)
    base = concatenate([base[features], y_out_of_c], axis=1)

    for realization in range(20):
        train, test = split_random(base, train_percentage=.8)
        train, train_val = split_random(train, train_percentage=.8)

        x_train = train[:, :len(features)]
        y_train = train[:, len(features):]

        x_train_val = train_val[:, :len(features)]
        y_train_val = train_val[:, len(features):]

        x_test = test[:, :len(features)]
        y_test = test[:, len(features):]

        number_of_neourons = [10, 15, 20, 25, 30]
        simple_net = ExtremeLearningMachines(number_of_neurons=15, N_Classes=3, case='classification')
        simple_net.fit(x_train, y_train, x_train_val=x_train_val, y_train_val=y_train_val,
                       hidden=number_of_neourons)

        y_out = simple_net.predict(x_test, bias=True)
        y_test = simple_net.predicao(y_test)


        metrics_calculator = metric(y_test, y_out, types=['ACCURACY', 'precision', 'recall', 'f1_score'])
        metric_results = metrics_calculator.calculate(average='macro')
        print(metric_results)

        results['cf'].append((metric_results['ACCURACY'],
                              metrics_calculator.confusion_matrix(list(y_test), y_out, labels=list(range(C))),
                              simple_net.number_of_neurons
                              ))


        results['realization'].append(realization)
        for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
            results[type].append(metric_results[type])

    results['cf'].sort(key=lambda x: x[0], reverse=True)
    final_result['best_cf'].append(results['cf'][0][1])
    best_number_neurons = results['cf'][0][2]


    for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
        final_result[type].append(mean(results[type]))
        final_result['std ' + type].append(std(results[type]))



    # ------------------------ PLOT -------------------------------------------------
    #
    for i in range(len(final_result['best_cf'])):
        plt.figure(figsize=(10, 7))

        df_cm = DataFrame(final_result['best_cf'][i], index=[i for i in range(C)],
                             columns=[i for i in range(C)])
        sn.heatmap(df_cm, annot=True)
        plt.title('Matriz de connfusão dermatologia com número de neurônios: ' + str(best_number_neurons))
        plt.xlabel('Valor Esperado')
        plt.ylabel('Valor Encontrado')

        path = get_project_root() + '/run/TR-06/CANCER/results/'
        plt.savefig(path + "mat_confsuison_elm.jpg")
        plt.show()

    print(pd.DataFrame(final_result))
    # del final_result['best_cf']
    pd.DataFrame(final_result).to_csv(get_project_root() + '/run/TR-06/CANCER/results/' + 'result_elm.csv')
