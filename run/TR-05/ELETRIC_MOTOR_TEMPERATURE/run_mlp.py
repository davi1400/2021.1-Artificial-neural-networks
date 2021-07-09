import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
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
from mlfwk.models import MultiLayerPerceptron
from mlfwk.visualization import generate_space, coloring

if __name__ == '__main__':
    print("run emt")

    # --------------------------- Read dataset ----------------------------------------

    df = load_base('measures_v2.csv', type='csv')
    df = df.drop(['profile_id'], axis=1)

    df = df.iloc[:100000]
    nRow, nCol = df.shape
    print(f'There are {nRow} rows and {nCol} columns')


    df.info()

    features = ['u_q', 'coolant', 'u_d', 'motor_speed',
                'i_d', 'i_q', 'ambient', 'torque'] # 'profile_id'

    targets = ['stator_yoke', 'pm',  'stator_winding', 'stator_tooth']

    # -------------------- Realiztions ---------------------------------------------

    # normalizar a base
    df[features] = normalization(df[features], type='min-max')

    N, M = df.shape
    C = 1  # Problema de regressão

    for different_target in targets:
        print('Target: ' + different_target)
        final_result = {
            'MSE': [],
            'std MSE': [],
            'RMSE': [],
            'std RMSE': [],
            'alphas': []
        }

        results = {
            'realization': [],
            'MSE': [],
            'RMSE': [],
            'alphas': []
        }
        for realization in range(5):
            train, test = split_random(df, train_percentage=.8)
            # train, train_val = split_random(train, train_percentage=.8)

            # ------------------------------ x and y for training -----------------------------------
            x_train = train[features]
            y_train = train[different_target].to_numpy().reshape(train[different_target].shape[0], 1)

            # ------------------------------ x and y for validation -----------------------------------

            # x_train_val = train_val[features]
            # y_train_val = train_val[different_target]

            # ------------------------------ x and y for test ------------------------------------------

            x_test = test[features]
            y_test = test[different_target]
            y_test.to_numpy().reshape(y_test.shape[0], 1)

            # ---------------------------------- modeling ----------------------------------------------
            validation_alphas = [0.15]
            hidden = 4 * np.arange(1, 5)
            simple_net = MultiLayerPerceptron(9, C, epochs=1000, Regressao=True, hidden_layer_neurons=12, learning_rate=0.15)
            simple_net.fit(x_train.to_numpy(), y_train,
                           x_train_val=[], y_train_val=[],
                           alphas=validation_alphas,
                           hidden=hidden,
                           validation=False)

            y_out = simple_net.predict(x_test, bias=True)

            metrics_calculator = metric(y_test, y_out, types=['MSE', 'RMSE'])
            metric_results = metrics_calculator.calculate()
            print(metric_results)

            results['alphas'].append(simple_net.lr)
            results['realization'].append(realization)
            for type in ['MSE', 'RMSE']:
                results[type].append(metric_results[type])

        final_result['alphas'].append(mean(results['alphas']))
        for type in ['MSE', 'RMSE']:
            final_result[type].append(mean(results[type]))
            final_result['std ' + type].append(std(results[type]))

        print(pd.DataFrame(final_result))
        pd.DataFrame(final_result).to_csv(
            get_project_root() + '/run/TR-05/ELETRIC_MOTOR_TEMPERATURE/results/' + 'result_mlp_' + different_target + '.csv')


    # ------------------------------------------------------------------------------------




