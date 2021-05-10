import sys
from pathlib import Path

print(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import seaborn as sn
import matplotlib.pyplot as plt

from mlfwk.readWrite import load_mock
from mlfwk.models import adaline
from mlfwk.metrics import metric
from numpy import where, mean, std, c_, array, linspace, concatenate, meshgrid
from pandas.core.frame import DataFrame
from mlfwk.utils import split_random, get_project_root
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

    F, x1, x2 = load_mock(type='2D_REGRESSOR')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, F, color='k')
    ax.set_title("y = ax1 + bx2 + c")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    plt.savefig(get_project_root() + '/run/TR-02/ARTIFICIAL/results/' + 'adaline_fig_2.jpg')
    plt.show()

    base = concatenate([array(x1, ndmin=2), array(x2, ndmin=2), array(F, ndmin=2)], axis=0).T
    validation_alphas = linspace(0.015, 0.1, 20)

    for realization in range(20):
        train, test = split_random(base, train_percentage=.8)
        train, train_val = split_random(train, train_percentage=0.7)

        x_train = train[:, :2]
        y_train = train[:, 2]

        x_train_val = train_val[:, :2]
        y_train_val = train_val[:, 2]

        x_test = test[:, :2]
        y_test = test[:, 2]

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

    final_result['alphas'].append(mean(results['alphas']))
    for type in ['MSE', 'RMSE']:
        final_result[type].append(mean(results[type]))
        final_result['std ' + type].append(std(results[type]))

    results['erros'].sort(key=lambda x: x[0], reverse=True)
    min_error = results['erros'][0]
    plt.plot(list(range(len(min_error[1]))), min_error[1])
    plt.xlabel("epocas")
    plt.ylabel("MSE")
    plt.ylim(0, 0.2)
    plt.savefig(get_project_root() + '/run/TR-02/ARTIFICIAL/results/' + "MSE_X_EPOCH_artifical_2.jpg")
    plt.plot()

    from matplotlib import cm

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')



    # ax.plot_surface(xx, yy, z, rstride=1, cstride=1, alpha=0.2)
    # ax.scatter(x1, x2, F, c='blue')
    # ax.set_title("g = ax1 + bx2 + c")
    # ax.set_xlabel("x1")
    # ax.set_ylabel("x2")
    # ax.set_zlabel("y")
    # ax.axis('tight')
    # plt.savefig(get_project_root() + '/run/TR-02/ARTIFICIAL/results/' + 'result_adaline_fig_2.jpg')
    # plt.show()

    # --------------------------------------- Predicting the surface ---------------------------------#

    w = regressor_adaline.__coef__()
    xx, yy = meshgrid(x1, x2)
    z = w[1] * xx + w[2] * yy + w[0]

    # ------------------------------------------- Plot --------------------------------------------------------#

    plt.style.use('default')

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    axes = [ax1, ax2, ax3]

    for ax in axes:
        ax.plot(x1, x2, F, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5, label='Conjunto de dados')
        ax.scatter(xx.flatten(), yy.flatten(), z, facecolor=(0, 0, 0, 0), s=20, edgecolor='#70b3f0', label='g(x)')
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('x2', fontsize=12)
        ax.set_zlabel('y', fontsize=12)
        ax.locator_params(nbins=4, axis='x')
        ax.locator_params(nbins=5, axis='x')

    ax1.text2D(0.2, 0.32, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
               transform=ax1.transAxes, color='grey', alpha=0.5)
    ax2.text2D(0.3, 0.42, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
               transform=ax2.transAxes, color='grey', alpha=0.5)
    ax3.text2D(0.85, 0.85, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
               transform=ax3.transAxes, color='grey', alpha=0.5)

    ax1.view_init(elev=28, azim=120)
    ax2.view_init(elev=4, azim=114)
    ax3.view_init(elev=60, azim=165)

    # fig.suptitle('')

    fig.tight_layout()
    plt.legend()
    plt.savefig(get_project_root() + '/run/TR-02/ARTIFICIAL/results/' + 'result_adaline_fig_2.jpg')
    plt.show()

    print(DataFrame(final_result))
    DataFrame(final_result).to_csv(get_project_root() + '/run/TR-02/ARTIFICIAL/results/' + 'result_adaline_2.csv')
