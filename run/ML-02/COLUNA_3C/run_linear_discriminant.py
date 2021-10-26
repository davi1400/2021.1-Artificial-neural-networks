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
from mlfwk.models import linearDiscriminant
from mlfwk.readWrite import load_base
from mlfwk.visualization import generate_space, coloring


if __name__ == '__main__':
    print("run coluna 2 classes")
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
        'cf': [],
    }

    # carregar a base
    base = load_base(path='column_3C_weka.arff', type='arff')


    # features
    features = ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis']
    classes = base['class'].unique()

    print(base.info())

    # ----------------------------- Clean the data ----------------------------------------------------------------

    # -------------------------- Normalization ------------------------------------------------------------------

    # normalizar a base
    base[features] = normalization(base[features], type='min-max')


    # ------------------------------------------------------------------------------------------------------------

    for realization in range(20):
        train, test = split_random(base, train_percentage=.8)

        linear_clf = linearDiscriminant(classes, features, 'class')
        linear_clf.fit(train)
        y_out = linear_clf.predict(test)


        # decoding the types of outputs for calculate de the metrics
        y_test = list(test['class'])
        for i in range(len(y_out)):
            y_out[i] = y_out[i].decode()
            y_test[i] = y_test[i].decode()

        metrics_calculator = metric(y_test, y_out, types=['ACCURACY', 'precision', 'recall', 'f1_score'])
        metric_results = metrics_calculator.calculate(average='macro')
        print(metric_results)

        results['cf'].append(
            (metric_results['ACCURACY'], metrics_calculator.confusion_matrix(y_test, y_out,
                                                                             labels=['Hernia', 'Spondylolisthesis', 'Normal'])))

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

        df_cm = DataFrame(final_result['best_cf'][i], index=[i for i in ['Hernia', 'Spondylolisthesis', 'Normal']],
                             columns=[i for i in ['Hernia', 'Spondylolisthesis', 'Normal']])
        sn.heatmap(df_cm, annot=True)

        path = get_project_root() + '/run/ML-02/COLUNA_3C/results/'
        plt.savefig(path + "mat_confsuison_triangle_LD.jpg")
        plt.show()

    print(pd.DataFrame(final_result))
    pd.DataFrame(final_result).to_csv(get_project_root() + '/run/ML-02/COLUNA_3C/results/' + 'result_LD.csv')
