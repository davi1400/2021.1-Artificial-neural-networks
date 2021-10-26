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
from mlfwk.models import gaussianBayes
from mlfwk.readWrite import load_base
from mlfwk.visualization import generate_space, coloring

if __name__ == '__main__':
    print("run breast cancer")
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
    classes = base['Class'].unique()

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
    for realization in range(20):
        train, test = split_random(base, train_percentage=.8)

        bayes_gaussian_clf = gaussianBayes(classes)
        bayes_gaussian_clf.fit(train, features, 'Class')
        y_out = bayes_gaussian_clf.predict(test, features)

        metrics_calculator = metric(list(test['Class']), y_out, types=['ACCURACY', 'precision', 'recall', 'f1_score'])
        metric_results = metrics_calculator.calculate(average='macro')
        print(metric_results)

        results['cf'].append((metric_results['ACCURACY'],
                              metrics_calculator.confusion_matrix(list(test['Class']), y_out, labels=classes)))

        results['realization'].append(realization)
        for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
            results[type].append(metric_results[type])

    results['cf'].sort(key=lambda x: x[0], reverse=True)
    final_result['best_cf'].append(results['cf'][0][1])
    for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
        final_result[type].append(mean(results[type]))
        final_result['std ' + type].append(std(results[type]))



    # ------------------------ PLOT -------------------------------------------------
    #
    for i in range(len(final_result['best_cf'])):
        plt.figure(figsize=(10, 7))

        df_cm = DataFrame(final_result['best_cf'][i], index=[i for i in classes],
                             columns=[i for i in classes])
        sn.heatmap(df_cm, annot=True)

        path = get_project_root() + '/run/ML-01/CANCER/results/'
        plt.savefig(path + "mat_confsuison.jpg")
        plt.show()

    print(pd.DataFrame(final_result))
    pd.DataFrame(final_result).to_csv(get_project_root() + '/run/ML-01/CANCER/results/' + 'result_bayes_gaussian.csv')
