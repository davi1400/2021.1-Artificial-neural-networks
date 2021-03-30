from mlfwk.algorithms import calculate_euclidian_distance


class knn:
    def __init__(self, x, y, k=3):
        self.x = x
        self.y = y
        self.k = k

    def predict(self, x_test):
        self.running()

        y_output = []
        for exampe in x_test:
            distance = []
            for indice in range(len(self.x)):
                euclidian_dist = calculate_euclidian_distance(exampe, self.x[indice])
                distance.append((euclidian_dist, self.y[indice]))

            k_nearest_distances_and_labels = sorted(distance)[:self.k]
            counter = {}
            for i in range(len(k_nearest_distances_and_labels)):
                label = str(int(k_nearest_distances_and_labels[i][1][0]))

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


        pass

    def running(self):
        """Begin to run"""
        return


if __name__ == '__main__':
    pass
