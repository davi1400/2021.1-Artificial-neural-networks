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
from mlfwk.models import RadialBasisFunction
from mlfwk.visualization import generate_space, coloring

if __name__ == '__main__':
    print("run emt")

    # --------------------------- Read dataset ----------------------------------------

    df = load_base('measures_v2.csv', type='csv')
    df = df.drop(['profile_id'], axis=1)

    # df = df.iloc[:1000]
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
            'R2': [],
            'std R2': []
        }

        results = {
            'realization': [],
            'MSE': [],
            'RMSE': [],
            'R2': []
        }
        for realization in range(5):
            train, test = split_random(df, train_percentage=.8)
            train, train_val = split_random(train, train_percentage=.8)

            # ------------------------------ x and y for training -----------------------------------
            x_train = train[features]
            y_train = train[different_target].to_numpy().reshape(train[different_target].shape[0], 1)

            # ------------------------------ x and y for validation -----------------------------------

            x_train_val = train_val[features]
            y_train_val = train_val[different_target]

            # ------------------------------ x and y for test ------------------------------------------

            x_test = test[features]
            y_test = test[different_target]
            y_test.to_numpy().reshape(y_test.shape[0], 1)

            # ---------------------------------- modeling ----------------------------------------------
            validation_alphas = np.linspace(1.0, 3.0, 10)
            hidden = 3 * np.arange(1, 10)
            simple_net = RadialBasisFunction(number_of_neurons=30, N_Classes=1, alpha=3.5, case='regression')
            simple_net.fit(x_train, y_train, x_train_val=x_train_val, y_train_val=y_train_val, alphas=validation_alphas,
                           hidden=hidden, validation=False)

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
        pd.DataFrame(final_result).to_csv(
            get_project_root() + '/run/TR-06/ELETRIC_MOTOR_TEMPERATURE/results/' + 'result_rbf_' + different_target + '.csv')


    # ------------------------------------------------------------------------------------




