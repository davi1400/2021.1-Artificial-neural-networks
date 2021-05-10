import sys
from pathlib import Path

print(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import seaborn as sn
import matplotlib.pyplot as plt

from mlfwk.readWrite import load_mock
from mlfwk.models import adaline
from mlfwk.metrics import metric
from numpy import where, mean, std, c_, array, linspace, concatenate
from pandas.core.frame import DataFrame
from mlfwk.utils import split_random, get_project_root

if __name__ == '__main__':

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
        'erros': [],
        'alphas': []
    }

    F, x = load_mock(type='LINEAR_REGRESSOR')

    # plt.plot(x, F, 'bo', color='k')
    # # plt.plot(array(x, ndmin=2).T, regressor_adaline.predict(array(x, ndmin=2).T))
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.savefig(get_project_root() + '/run/TR-02/ARTIFICIAL/results/' + 'adaline_fig_1.jpg')
    # plt.show()

    plt.style.use('default')
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(8, 4))

    # ax.plot(array(x, ndmin=2).T, regressor_adaline.predict(array(x, ndmin=2).T), color='k', label='g(x)')
    ax.scatter(x, F, edgecolor='k', facecolor='grey', alpha=0.7, label='Conjunto de dados')
    ax.set_ylabel('F', fontsize=14)
    ax.set_xlabel('x1', fontsize=14)
    ax.legend(facecolor='white', fontsize=11)
    ax.set_title('y = ax1 + b', fontsize=18)

    fig.tight_layout()
    plt.savefig(get_project_root() + '/run/TR-02/ARTIFICIAL/results/' + 'adaline_fig_1.jpg')
    plt.show()


    base = concatenate([array(x, ndmin=2), array(F, ndmin=2)], axis=0).T
    validation_alphas = linspace(0.015, 0.1, 20)
    for realization in range(20):
        train, test = split_random(base, train_percentage=.8)
        train, train_val = split_random(train, train_percentage=0.7)

        x_train = train[:, :1]
        y_train = train[:, 1]

        x_train_val = train_val[:, :1]
        y_train_val = train_val[:, 1]

        x_test = test[:, :1]
        y_test = test[:, 1]

        regressor_adaline = adaline(epochs=1000, learning_rate=0.01)

        regressor_adaline.fit(x_train, y_train, x_train_val,
                              y_train_val, alphas=validation_alphas)

        y_out_adaline = regressor_adaline.predict(x_test)

        metrics_calculator = metric(list(y_test), y_out_adaline, types=['MSE', 'RMSE'])
        metric_results = metrics_calculator.calculate()

        results['erros'].append((metric_results['MSE'], regressor_adaline.errors_per_epoch))
        results['alphas'].append(regressor_adaline.learning_rate)
        results['realization'].append(realization)
        for type in ['MSE', 'RMSE']:
            results[type].append(metric_results[type])

    results['erros'].sort(key=lambda x: x[0], reverse=True)
    min_error = results['erros'][0]
    plt.plot(list(range(len(min_error[1]))), min_error[1])
    plt.xlabel("epocas")
    plt.ylabel("MSE")
    plt.ylim(0, 0.2)
    plt.savefig(get_project_root() + '/run/TR-02/ARTIFICIAL/results/' + "MSE_X_EPOCH_artifical_1.jpg")
    plt.plot()


    final_result['alphas'].append(mean(results['alphas']))
    for type in ['MSE', 'RMSE']:
        final_result[type].append(mean(results[type]))
        final_result['std ' + type].append(std(results[type]))

    # plt.plot(x, F, 'bo')
    # plt.plot(array(x, ndmin=2).T, regressor_adaline.predict(array(x, ndmin=2).T), color='k', label="g(x)")
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.savefig(get_project_root() + '/run/TR-02/ARTIFICIAL/results/' + 'result_adaline_fig_1.jpg')
    # plt.show()

    # ############################################# Plot ############################################### #

    plt.style.use('default')
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(array(x, ndmin=2).T, regressor_adaline.predict(array(x, ndmin=2).T), color='k', label='g(x)')
    ax.scatter(x, F, edgecolor='k', facecolor='grey', alpha=0.7, label='Conjunto de dados')
    ax.set_ylabel('F', fontsize=14)
    ax.set_xlabel('x1', fontsize=14)
    # ax.text(0.8, 0.1, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
    #         transform=ax.transAxes, color='grey', alpha=0.5)
    ax.legend(facecolor='white', fontsize=11)
    # ax.set_title('$R^2= %.2f$' % r2, fontsize=18)

    fig.tight_layout()
    plt.savefig(get_project_root() + '/run/TR-02/ARTIFICIAL/results/' + 'result_adaline_fig_1.jpg')
    plt.show()

    print(DataFrame(final_result))
    DataFrame(final_result).to_csv(get_project_root() + '/run/TR-02/ARTIFICIAL/results/' + 'result_adaline_1.csv')
