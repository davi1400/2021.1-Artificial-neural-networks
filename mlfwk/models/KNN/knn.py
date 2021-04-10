from numpy import array
from pandas.core.frame import DataFrame
from mlfwk.algorithms import calculate_euclidian_distance


class knn:
    def __init__(self, x_train, y_train, class_column_name=None, k=3):


        self.x_train = x_train
        self.y_train = y_train
        self.k = k

    def predict(self, x_test):
        self.running.__doc__

        y_output = []
        for example in x_test:
            distance = []
            for indice in range(len(self.x_train)):
                euclidian_dist = calculate_euclidian_distance(example, self.x_train[indice])
                distance.append((euclidian_dist, self.y_train[indice]))

            k_nearest_distances_and_labels = sorted(distance)[:self.k]
            counter = {}
            for i in range(len(k_nearest_distances_and_labels)):
                label = (int(k_nearest_distances_and_labels[i][1]))

                if label in counter:
                    counter[label] += 1
                else:
                    counter.update({label: 1})

            max_one = max(list(counter.values()))
            for label in counter.keys():
                if counter[label] == max_one:
                    y_output.append(label)
                    break

        return y_output

    def running(self):
        """Begin to run"""
        return


