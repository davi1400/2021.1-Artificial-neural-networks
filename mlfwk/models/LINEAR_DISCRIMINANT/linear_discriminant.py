from numpy import max, where, array, argmax, dot, log
from scipy import linalg


class linearDiscriminant:
    def __init__(self, classes, features, target, cov_type='all'):
        """

        @param classes:
        """
        self.classes = classes
        self.features = features
        self.target = target
        self.cov_type = cov_type
        self.prior = {}

    def fit(self, train):
        """

        @param train:
        @return:
        """

        self._calculate_prior_probability(train)
        self._get_mean_covariance(train)

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

        equation_part_one = (1 / 2) * dot(dot(x, self.cov_inv), self.mean_from_class[c])

        auxiliar = dot(self.cov_inv, self.mean_from_class[c])

        equation_part_two = (-1 / 2) * dot(self.mean_from_class[c].T, auxiliar)
        equation_part_three = (1 / 2) * dot(array(x, ndmin=2), auxiliar)

        discriminant = equation_part_one + equation_part_two + equation_part_three + log(self.prior[c])

        return discriminant[0][0]

    def _calculate_prior_probability(self, train):
        """

        @param train:
        @return:
        """
        N, M = train.shape  # pandas type (N examples and M features)
        for c in self.classes:
            self.prior.update({c: sum(train[self.target] == c) / N})

    def _get_mean_covariance(self, train):
        """

        @param train:
        @param features:
        @param class_column:
        @return:
        """

        self.mean_from_class = {}
        self.covariance = None

        for c in self.classes:
            self.mean_from_class.update(
                {c: array(train[train[self.target] == c][self.features].mean(), ndmin=2).T})  # (M, 1)

            if self.cov_type == 'pool':
                # covariance will be a mean from all
                part_one = (array(train[train[self.target] == c][self.features]).T - self.mean_from_class[c])
                part_two = (array(train[train[self.target] == c][self.features]).T - self.mean_from_class[c]).T
                self.covariance += self.prior[c] * (dot(part_one, part_two) / len(train[train[self.target] == c]))

        if self.cov_type != 'pool':
            # covariance from all data
            part_one = (array(train[self.features]).T - array(train[self.features].mean(), ndmin=2).T)
            part_two = (array(train[self.features]).T - array(train[self.features].mean(), ndmin=2).T).T

            self.covariance = dot(part_one, part_two) / (len(train) - 1)

        self.cov_inv = linalg.inv(self.covariance)
