import threading

import numpy as np
import pandas as pd
from numpy.random import rand, permutation
from numpy import where, append, ones, array, zeros, mean, argmax, linspace, concatenate, c_, argmin
from mlfwk.algorithms import calculate_euclidian_distance, calculate_centroid


class kmeans:
    def __init__(self, k=None, epsilon=.6, max_iter=300):
        self.k = k
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.centroids = None
        self.__before_centroids = None

    def fit(self, x, y, x_train_val=None, y_train_val=None, K=None, validation=True):
        if validation:
            self.k_fold_cross_validate(x_train_val, y_train_val, number_of_centers=K)

        return self.train(x, y)

    def k_fold_cross_validate(self, x, y, number_of_centers=None):
        validation_accuracys = []

        y = array(y, ndmin=2)
        for n in number_of_centers:
            K = 10
            k_validation_accuracys = []
            for esimo in range(1, K + 1):
                L = int(x.shape[0] / K)
                x_train_val = (c_[x[:L * esimo - L, :].T, x[esimo * L:, :].T]).T
                x_test_val = (x[L * esimo - L:esimo * L, :])
                y_train_val = (c_[y[:L * esimo - L, :].T, y[esimo * L:, :].T]).T
                y_test_val = (y[L * esimo - L:esimo * L, :])

                classifier = kmeans(k=n)

                classifier.fit(x_train_val, y_train_val, validation=False, bias=False)
                y_out_val = classifier.predict(x_test_val, test=False)

                calculate_metrics = metric(y_test_val.tolist(), y_out_val.tolist(), types=['ACCURACY'])
                metric_results = calculate_metrics.calculate(average='macro')
                k_validation_accuracys.append(metric_results['ACCURACY'])

            validation_accuracys.append(mean(k_validation_accuracys))

        best_indice = argmax(validation_accuracys)
        self.k = number_of_centers[best_indice]

    def init_centroids(self, x):
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)

        self.centroids = x[indices[:self.k]]
        self.__before_centroids = None


    def predict(self, x):
        N = x.shape[0]
        output = zeros((N, 1))
        for i in range(N):
            distances = []
            for centroid in self.centroids:
                distances.append(calculate_euclidian_distance(x[i], centroid))

            output[i] = argmin(distances)
        return output

    def train(self, x_train, y_train):
        self.init_centroids(x_train)

        N, M = x_train.shape

        aux = 9999
        count = 0
        while aux > self.epsilon and count < self.max_iter:
            # Calculating the distance matrix of the points
            distance_matrix = zeros((x_train.shape[0], self.k))
            for i in range(N):
                x = np.array((x_train[i]), ndmin=2).T  # (px1)
                for j in range(len(self.centroids)):
                    distance_matrix[i][j] = calculate_euclidian_distance(x, self.centroids[j])

                indice_one = argmin(distance_matrix[i])
                distance_matrix[i] = zeros((1, len(self.centroids)))
                distance_matrix[i][indice_one] = 1

            self.__before_centroids = self.centroids.copy()
            for i in range(self.k):
                indices = np.where(distance_matrix.T[i] == 1)[0]
                if len(indices) == 0:
                    n = 1e-10
                else:
                    n = len(indices)
                self.centroids[i] = (1/n)*np.sum(x_train[indices], axis=0)

            d = 0
            for i in range(self.k):
                d += calculate_euclidian_distance(self.centroids[i].T, self.__before_centroids[i])

            aux = (1/self.k)*d
            count += 1
