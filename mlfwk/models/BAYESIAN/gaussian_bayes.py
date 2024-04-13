"""

Implementar o classificar bayesiano gaussiano. Nesta implementação,

considere que os vetores de média e matrizes de covariância são específicas para cada
classe.

Além disso, não faça simplificações nos discriminantes a fim de sejam
computadas diretamente as probabilidades a posteriori para cada uma das
classes.

"""

from numpy import max, where, array, argmax, dot
from mlfwk.algorithms import gaussianPDF


class gaussianBayes:
    def __init__(self, classes):
        """

        @param classes: array with the classes like ['setosa', ....]

        """
        self.classes = classes
        self.prior = {}


    def fit(self, train, features, class_column):
        self._calculate_prior_probability(train, class_column)
        self._get_mean_covariance_from_class(train, features, class_column)

    def predict(self, test, features):

        out = []
        for x in array(test[features]):
            posteriori_probabilitys = []
            for c in self.classes:
                posteriori_probabilitys.append(self._calculate_posteriori_probability(array(x,ndmin=2).T, c))

            out.append(self.classes[argmax(posteriori_probabilitys)])

        return out

    def _calculate_prior_probability(self, train, class_column):
        """

        @param y: array numpy of classes (N,1)
        @return:
        """
        N, M = train.shape  # pandas type (N examples and M features)
        for c in self.classes:
            self.prior.update({c: sum(train[class_column] == c)/N})

    def _get_mean_covariance_from_class(self, train, features, class_column):
        """

        @param features:
        @param class_column:
        @param train: pandas matrix with x and y, y its the last column
        @return:
        """

        self.mean_from_class = {}
        self.covariance_from_class = {}

        for c in self.classes:

            self.mean_from_class.update({c: array(train[train[class_column] == c][features].mean(), ndmin=2).T})  # (M, 1)

            part_one = (array(train[train[class_column] == c][features]).T - self.mean_from_class[c])
            part_two = (array(train[train[class_column] == c][features]).T - self.mean_from_class[c]).T

            self.covariance_from_class.update({c: dot(part_one, part_two) / len(train[train[class_column] == c])})

    def _calculate_likelihood_probability(self, x, c):
        """

        @param x:
        @return:
        """
        return gaussianPDF(x, self.mean_from_class[c], self.covariance_from_class[c])


    def _calculate_posteriori_probability(self, x, c):
        """

        @param x:
        @param c:
        @return:
        """
        return self._calculate_likelihood_probability(x, c) * self.prior[c]
