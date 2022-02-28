import copy

import numpy as np
import math
import random
import matplotlib.pyplot as plt

data = (
    [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
    [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
    [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
    [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
    [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369],
    [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459])


def e_dist(vec1, vec2):
    return math.sqrt((vec1[0] - vec2[0]) ** 2 + (vec1[1] - vec2[1]) ** 2)


def K_means(dataset, k):
    # 初始化k个簇
    n_cluster = []
    for i in range(k):
        temp = []
        n_cluster.append(temp)

    # 计算总样本数
    m = len(data)

    # 随机生成k个随机数
    k_random = random.sample(range(0, m), k)
    # 存储均值向量
    mean = []
    for i in range(k):
        mean.append(dataset[k_random[i]])
    flag = 1

    while flag:
        for i in range(m):
            # 初始化距离列表
            dist = []
            # 计算每个样本到均值向量的距离
            for j in range(k):
                dist.append(e_dist(dataset[i], mean[j]))
            # 返回最小距离的下标
            min_index = dist.index(min(dist))
            # 将该元素添加到对应簇类中
            if dataset[i] not in n_cluster[min_index]:
                n_cluster[min_index].append(dataset[i])
            # 如果该元素已经在其他簇类中，就将其删除
            for a in range(k):
                if a != min_index:
                    if dataset[i] in n_cluster[a]:
                        n_cluster[a].remove(dataset[i])
                print("n_cluster[", a, "]:", n_cluster[a])

        x = []
        y = []
        [x.append(n_cluster[0][i][0]) for i in range(len(n_cluster[0]))]
        [y.append(n_cluster[0][i][1]) for i in range(len(n_cluster[0]))]
        plt.scatter(x, y, s=30, c="red", marker=".")
        x = []
        y = []
        [x.append(n_cluster[1][i][0]) for i in range(len(n_cluster[1]))]
        [y.append(n_cluster[1][i][1]) for i in range(len(n_cluster[1]))]
        plt.scatter(x, y, s=30, c="blue", marker=".")
        x = []
        y = []
        [x.append(n_cluster[2][i][0]) for i in range(len(n_cluster[2]))]
        [y.append(n_cluster[2][i][1]) for i in range(len(n_cluster[2]))]
        plt.scatter(x, y, s=30, c="green", marker=".")

        plt.scatter(mean[0][0], mean[0][1], s=60, c="black", marker="d")
        plt.scatter(mean[1][0], mean[1][1], s=60, c="black", marker="d")
        plt.scatter(mean[2][0], mean[2][1], s=60, c="black", marker="d")
        plt.xlabel("Density")
        plt.ylabel("Sugar content")
        plt.title("K-means")
        plt.show()

        mean_update = []
        for i in range(k):
            sum0 = 0.0
            sum1 = 0.0
            if n_cluster[i]:
                for j in range(len(n_cluster[i])):
                    sum0 += n_cluster[i][j][0]
                    sum1 += n_cluster[i][j][1]
                mean_update.append([sum0 / len(n_cluster[i]), sum1 / len(n_cluster[i])])
        # print("mean", mean)
        # print("mean_update:", mean_update)

        if mean_update == mean:
            flag = 0
        else:
            mean = mean_update


def DBSCAN(dataset, minpts, epsilon):
    m = len(dataset)
    core_obj = []
    contain = []
    cluster = [[]]

    for i in range(m):
        contain.append([])
        for j in range(m):
            if e_dist(dataset[i], dataset[j]) <= epsilon:
                contain[i].append(j)
        if len(contain[i]) >= minpts:
            core_obj.append(i)

    no_visited = list(range(30))

    while len(core_obj) > 0:
        old_v = copy.deepcopy(no_visited)
        o = random.sample(core_obj, 1)[0]
        Q = []
        Q.append(o)
        no_visited.remove(o)
        while len(Q) > 0:
            q = Q[0]
            Q.remove(q)
            if len(contain[q]) >= minpts:
                temp = [x for x in contain[q] if x in no_visited]
                for x in temp:
                    Q.append(x)
                for x in temp:
                    no_visited.remove(x)
        for x in no_visited:
            old_v.remove(x)
        cluster.append(old_v)
        for x in old_v:
            if x in core_obj:
                core_obj.remove(x)

    for i in range(1,len(cluster)):
        print(cluster[i])


if __name__ == '__main__':
    k = 3
    # K_means(data, k)
    DBSCAN(data, 5, 0.11)
