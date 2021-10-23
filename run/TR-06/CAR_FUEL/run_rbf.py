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
    print("run car fuel")
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

    df = load_base('measurements.csv', type='csv')

    # ---------------------------------- cleaning data base --------------------------------------

    # NaN Columns
    new_df = df.drop(columns=['refill liters', 'refill gas', 'specials'])

    # specials_dummies = pd.get_dummies(new_df['specials'])

    # change E10 and SP98, for numerical
    new_df['gas_type'][new_df['gas_type'] == 'E10'] = int(0)
    new_df['gas_type'][new_df['gas_type'] == 'SP98'] = int(1)
    new_df['gas_type'] = new_df['gas_type'].astype('int')

    target = ['consume']
    features = ['distance', 'speed', 'temp_outside', 'gas_type', 'AC', 'rain', 'sun', 'temp_inside']

    for i in range(new_df.shape[0]):
        if ',' in new_df['distance'].iloc[i]:
            new_df['distance'].iloc[i] = float(new_df['distance'].iloc[i].replace(',', '.'))
        if ',' in new_df['consume'].iloc[i]:
            new_df['consume'].iloc[i] = float(new_df['consume'].iloc[i].replace(',', '.'))
        if not isinstance(new_df['temp_inside'].iloc[i], float) and (',' in new_df['temp_inside'].iloc[i]):
            new_df['temp_inside'].iloc[i] = float(new_df['temp_inside'].iloc[i].replace(',', '.'))
        if isinstance(new_df['temp_inside'].iloc[i], str):
            new_df['temp_inside'].iloc[i] = float(new_df['temp_inside'].iloc[i])

    new_df['distance'] = new_df['distance'].astype('int')
    new_df['consume'] = new_df['consume'].astype('int')

    new_df['temp_inside'] = new_df['temp_inside'].fillna(new_df['temp_inside'].mean())

    print(new_df.info())
    # ---------------------------------------------------------------------------------------------


    # normalizar a base
    new_df[features] = normalization(new_df[features], type='min-max')


    # --------------------------------------------------------------------------------------------

    N, M = new_df.shape
    C = 1  # Problema de regressão

    for realization in range(20):
        train, test = split_random(new_df, train_percentage=.8)
        train, train_val = split_random(train, train_percentage=.8)

        x_train = train[features]
        y_train = train[target]

        x_train_val = train_val[features]
        y_train_val = train_val[target]

        x_test = test[features]
        y_test = test[target]

        validation_alphas = [1.0, 1.5, 2.0]
        hidden = [10, 15, 20]
        simple_net = RadialBasisFunction(number_of_neurons=15, N_Classes=1, alpha=1.5, case='regression')
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
    pd.DataFrame(final_result).to_csv(
        get_project_root() + '/run/TR-06/CAR_FUEL/results/' + 'result_rbf.csv')


    #  ------------------------ PLOT -------------------------------------------------=
