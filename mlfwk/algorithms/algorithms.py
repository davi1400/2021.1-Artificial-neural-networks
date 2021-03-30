def calculate_euclidian_distance(X_example, example):
    dist = 0
    for i in range(len(X_example)):
        dist += (X_example[i] - example[i]) ** 2
    return dist
