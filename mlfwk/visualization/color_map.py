from mlfwk.readWrite import load_mock
from numpy import meshgrid, arange
import matplotlib.pyplot as plt
import seaborn as sns  # sits ontop of matplotlib


def generate_space(X):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = meshgrid(arange(x_min, x_max, .02), arange(y_min, y_max, .02))

    return xx, yy


def coloring(plot_dict, color_map, title=None, xlabel=None, ylabel=None, xlim=[], ylim=[], save=False, path=''):
    """

    @param plot_dict:
        :param plot_dict:  {
            'xx': meshgrid,
            'yy': meshgrid
            'Z': [y_1, y_2, ..., y_n]
            'classe_0': {
                    X: [x_1, x_2 .., x_n] 2-D
                    point: string
                    marker: string
                }
                .
                .
                .
            'classe_n':{
                    X: [x_1, x_2 .., x_n] 2-D
                    point: string
                    marker: string
                }
        }
        :return:
    @param color_map:
    @param title:
    @param xlabel:
    @param ylabel:
    @param xlim:
    @param ylim:
    @param save:
    @param path:
    @return:
    """
    plot_dict['Z'] = plot_dict['Z'].reshape(plot_dict['xx'].shape)
    plt.pcolormesh(plot_dict['xx'], plot_dict['yy'], plot_dict['Z'], cmap=color_map)
    for c in plot_dict['classes'].keys():
        plt.plot(plot_dict['classes'][c]['X'][:, 0], plot_dict['classes'][c]['X'][:, 1],
                 plot_dict['classes'][c]['point'], marker=plot_dict['classes'][c]['marker'], markeredgecolor='w')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(xlim) > 0:
        plt.xlim(xlim)
    if len(ylim) > 0:
        plt.ylim(ylim)
    plt.title(title)

    if save:
        plt.savefig(path)
    plt.show()


def subplots():
    pass
