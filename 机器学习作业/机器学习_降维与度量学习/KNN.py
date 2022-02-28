import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import ListedColormap

data = [[0.697, 0.460, 1],
        [0.774, 0.376, 1],
        [0.634, 0.264, 1],
        [0.608, 0.318, 1],
        [0.556, 0.215, 1],
        [0.430, 0.237, 1],
        [0.481, 0.149, 1],
        [0.437, 0.211, 1],
        [0.666, 0.091, 0],
        [0.243, 0.267, 0],
        [0.245, 0.057, 0],
        [0.343, 0.099, 0],
        [0.639, 0.161, 0],
        [0.657, 0.198, 0],
        [0.360, 0.370, 0],
        [0.593, 0.042, 0],
        [0.719, 0.103, 0]]

def l2_dist(vec1, vec2):
    return math.sqrt((vec1[0] - vec2[0]) ** 2 + (vec1[1] - vec2[1]) ** 2)


def knn(data, k):
    dist = [[]]
    for i in range(len(data)):
        dist.append([])
        for j in range(len(data)):
            dist[i].append(l2_dist(data[i], data[j]))

    dist_sort = [[]]
    for i in range(len(data)):
        dist_sort.append([])
        for j in range(len(data)):
            min_index = dist[i].index(min(dist[i]))
            dist_sort[i].append(min_index)
            dist[i][min_index] = 1.0
    label_pred = []

    for i in range(len(data)):
        sum = 0.0
        for j in range(k):
            sum += int(data[dist_sort[i][j]][2])
        label_pred.append(1 if sum>k/2.0 else 0)
    return label_pred


if __name__ == '__main__':
    k = 3
    y_pred = knn(data,k)
    print(y_pred)

