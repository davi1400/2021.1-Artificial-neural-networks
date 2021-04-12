import seaborn as sns
import matplotlib.pyplot as plt
from mlfwk.readWrite import load_base
from mlfwk.utils import get_project_root

if __name__ == '__main__':
    # create a figure "fig" with axis "ax1" with 3x2 configuration
    iris = load_base(path='iris.data', type='csv')
    fig, ax1 = plt.subplots(3, 2, sharex='col', figsize=(22, 18), gridspec_kw={'hspace': 0, 'wspace': 0.1})

    # 1st plot
    sns.set_style("whitegrid")
    sns.scatterplot(data=iris, x="SepalLengthCm", y="SepalWidthCm", hue="Species", ax=ax1[0, 0], legend='brief')

    # 2nd plot
    sns.scatterplot(data=iris, x="SepalWidthCm", y="SepalLengthCm", hue="Species", ax=ax1[0, 1], legend='brief')

    # 3rd plot
    sns.scatterplot(data=iris, x="SepalLengthCm", y="PetalLengthCm", hue="Species", ax=ax1[1, 0], legend='brief')

    # 4th plot
    sns.scatterplot(data=iris, x="SepalWidthCm", y="PetalLengthCm", hue="Species", ax=ax1[1, 1], legend='brief')

    # 5th
    sns.scatterplot(data=iris, x="SepalLengthCm", y="PetalWidthCm", hue="Species", ax=ax1[2, 0], legend='brief')

    # 6th
    sns.scatterplot(data=iris, x="SepalWidthCm", y="PetalWidthCm", hue="Species", ax=ax1[2, 1], legend='brief')

    fig.savefig(get_project_root() + '/run/TR-00/IRIS/results/' + "Iris.jpg")
