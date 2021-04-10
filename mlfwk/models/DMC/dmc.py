from numpy import where, zeros, argmin
from mlfwk.algorithms import calculate_centroid, calculate_euclidian_distance


class dmc:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        # self.k = k

    def find_centroids(self, classes):
        """

        :param classes:
        :return:
        """

        self.centroids_per_class = {}

        for c in classes:
            indices = where(self.y_train == c)[0]

            self.centroids_per_class.update({
                c: calculate_centroid(self.x_train[indices])
            })
        print(self.centroids_per_class)
        
    def predict(self, x_test, classes):
        self.running.__doc__
        self.find_centroids(classes)

        y_output = []
        for example in x_test:
            i = 0
            distance = zeros((len(classes), len(classes)))
            for c in self.centroids_per_class.keys():
                euclidian_dist = calculate_euclidian_distance(example, self.centroids_per_class[c])
                distance[i] = [euclidian_dist, c]
                i += 1

            i_min_distance = argmin(distance[:, 0])
            y_output.append(distance[i_min_distance][1])

        return y_output

    def validation(self):
        pass

    def running(self):
        return
