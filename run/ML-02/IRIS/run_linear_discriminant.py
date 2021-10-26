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
from mlfwk.readWrite import load_base
from mlfwk.visualization import generate_space, coloring

if __name__ == '__main__':
    print("run iris")
    final_result = {
        'ACCURACY': [],
        'std ACCURACY': [],
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
        'f1_score': [],
        'precision': [],
        'recall': [],
    }

    # carregar a base
    base = load_base(path='iris.data', type='csv')

    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


    # normalizar a base
    base[features] = normalization(base[features], type='min-max')
    classes = base['Species'].unique()


    for realization in range(20):
        train, test = split_random(base, train_percentage=.8)

        linear_clf = linearDiscriminant(classes, features, 'Species')
        linear_clf.fit(train)
        y_out = linear_clf.predict(test)


        metrics_calculator = metric(list(test['Species']), y_out, types=['ACCURACY', 'precision', 'recall', 'f1_score'])
        metric_results = metrics_calculator.calculate(average='macro')
        print(metric_results)


        results['realization'].append(realization)
        for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
            results[type].append(metric_results[type])

    for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
        final_result[type].append(mean(results[type]))
        final_result['std ' + type].append(std(results[type]))


    print(pd.DataFrame(final_result))
    pd.DataFrame(final_result).to_csv(get_project_root() + '/run/ML-02/IRIS/results/' + 'result_linear_discriminant.csv')
