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
from mlfwk.models import RadialBasisFunction
from mlfwk.visualization import generate_space, coloring

if __name__ == '__main__':
    print("run artificial seno")
    final_result = {
        'MSE': [],
        'std MSE': [],
        'RMSE': [],
        'std RMSE': [],
        'R2': [],
        'std R2': []
    }

    results = {
        'realization': [],
        'MSE': [],
        'RMSE': [],
        'R2': []
    }

    base = load_mock(type='MOCK_SENO')
    # normalizar a base
    base[['x1']] = normalization(base[['x1']], type='min-max')

    sn.set_style('whitegrid')
    sn.scatterplot(data=base, x="x1", y="y", color='c')
    plt.xlabel("X1")
    plt.ylabel("Y")
    plt.savefig(get_project_root() + '/run/TR-05/ARTIFICIAL_REGRESSAO/results/' + 'dataset_seno_artificial.png')
    plt.show()

    x = array(base[['x1']])
    y = array(base[['y']])


    N, M = base.shape
    C = 1  # Problema de regressão

    epochs = 2000
    for realization in range(20):
        train, test = split_random(base, train_percentage=.8)
        train, train_val = split_random(train, train_percentage=.8)

        x_train = train[['x1']]
        y_train = train[['y']]

        x_train_val = train_val[['x1']]
        y_train_val = train_val[['y']]

        x_test = test[['x1']]
        y_test = test[['y']]

        validation_alphas = [0.1, 0.3, 0.5, 1.0]
        hidden = [10, 15, 20]
        simple_net = RadialBasisFunction(number_of_neurons=20, N_Classes=1, alpha=0.1, case='regression')
        simple_net.fit(x_train, y_train, x_train_val=x_train_val, y_train_val=y_train_val, alphas=validation_alphas,
                       hidden=hidden)

        y_out = simple_net.predict(x_test, bias=True)

        metrics_calculator = metric(y_test, y_out, types=['MSE', 'RMSE', 'R2'])
        metric_results = metrics_calculator.calculate()
        print(metric_results)


        results['realization'].append(realization)
        for type in ['MSE', 'RMSE', 'R2']:
            results[type].append(metric_results[type])


    for type in ['MSE', 'RMSE', 'R2']:
        final_result[type].append(mean(results[type]))
        final_result['std ' + type].append(std(results[type]))

    print(pd.DataFrame(final_result))
    pd.DataFrame(final_result).to_csv(get_project_root() + '/run/TR-06/ARTIFICIAL_REGRESSAO/results/' + 'result_rbf.csv')

    # # ------------------------ PLOT -------------------------------------------------

    plt.plot(x, simple_net.predict(x, bias=True))
    sn.set_style('whitegrid')
    sn.scatterplot(data=base, x="x1", y="y", color='c')
    plt.xlabel("X1")
    plt.ylabel("Y")
    plt.savefig(get_project_root() + '/run/TR-06/ARTIFICIAL_REGRESSAO/results/' + 'result_rbf_seno.png')
    plt.show()