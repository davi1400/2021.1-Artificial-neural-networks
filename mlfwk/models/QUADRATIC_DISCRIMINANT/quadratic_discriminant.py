from numpy import max, where, array, argmax, dot, log
from scipy import linalg


class quadraticDiscriminant:
    def __init__(self, classes, features, target):
        """

        @param classes:
        """
        self.classes = classes
        self.features = features
        self.target = target
        self.prior = {}

    def fit(self, train):
        """

        @param train:
        @return:
        """

        self._calculate_prior_probability(train)
        self._get_mean_covariance_from_class(train)

    def predict(self, test):
        """

        @param test:
        @return:
        """
        out = []
        for x in array(test[self.features]):
            discrimants = []  # array contain each discriminan value for each class
            for c in self.classes:
                discrimants.append(self._calculate_discriminant_for_class(x, c))

            # get the discriminant with the greatest value
            out.append(self.classes[argmax(discrimants)])

        return out

    def _calculate_discriminant_for_class(self, x, c):
        """

        @param x:
        @param c:
        @return:
        """

        self.cov_inv_for_class = linalg.inv(self.covariance_from_class[c])

        quadractic_part = (-1 / 2) * dot(dot(x, self.cov_inv_for_class), x)

        equation_part_two = (1 / 2) * dot(dot(x, self.cov_inv_for_class), self.mean_from_class[c])

        auxiliar = dot(self.cov_inv_for_class, self.mean_from_class[c])

        equation_part_three = (-1 / 2) * dot(self.mean_from_class[c].T, auxiliar)
        equation_part_four = (1 / 2) * dot(array(x, ndmin=2), auxiliar)


        discriminant = quadractic_part + equation_part_two + equation_part_three + equation_part_four + log(
            self.prior[c])

        return discriminant[0][0]

    def _calculate_prior_probability(self, train):
        """

        @param train:
        @return:
        """
        N, M = train.shape  # pandas type (N examples and M features)
        for c in self.classes:
            self.prior.update({c: sum(train[self.target] == c) / N})

    def _get_mean_covariance_from_class(self, train):
        """

        @param features:
        @param class_column:
        @param train: pandas matrix with x and y, y its the last column
        @return:
        """

        self.mean_from_class = {}
        self.covariance_from_class = {}

        for c in self.classes:
            self.mean_from_class.update(
                {c: array(train[train[self.target] == c][self.features].mean(), ndmin=2).T})  # (M, 1)

            part_one = (array(train[train[self.target] == c][self.features]).T - self.mean_from_class[c])
            part_two = (array(train[train[self.target] == c][self.features]).T - self.mean_from_class[c]).T

            self.covariance_from_class.update({c: dot(part_one, part_two) / len(train[train[self.target] == c])})
