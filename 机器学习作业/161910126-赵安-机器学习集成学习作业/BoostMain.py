from matplotlib.pyplot import MultipleLocator
from sklearn import tree
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import math


def adaboost(x_train, y_train, x_test, y_test, M):
    n_train = x_train.shape[0]
    pred_train = np.zeros(n_train)

    n_test = x_test.shape[0]
    pred_test = np.zeros(n_test)

    # 定义分类器
    clf = tree.DecisionTreeClassifier(max_depth=2)
    # 初始化数据权重
    w = np.ones(n_train) / n_train
    # 初始化模型权重
    theta = np.zeros(M)
    # 循环迭代
    pred_train = np.zeros(n_train)
    test_auc = 0
    for i in range(M):
        # 训练一个弱分类器
        clf.fit(x_train, y_train, sample_weight=w)
        # 计算弱分类器误差
        pred_train_i = clf.predict(x_train)
        pred_test_i = clf.predict(x_test)
        miss = [int(x) for x in (pred_train_i != y_train)]
        error = np.dot(w, miss)
        # 计算弱分类器的权重
        theta[i] = 0.5 * np.log((1 - error) / float(error))
        # 更新数据权重
        miss2 = [1 if x == 1 else -1 for x in miss]
        w = np.multiply(w, np.exp([float(x) * theta[i] for x in miss2]))
        w = w / sum(w)

        pred_train_i = [1 if x == 1 else -1 for x in pred_train_i]
        pred_test_i = [1 if x == 1 else -1 for x in pred_test_i]
        pred_train = pred_train + np.multiply(theta[i], pred_train_i)
        pred_test = pred_test + np.multiply(theta[i], pred_test_i)
        if i == 30:
            print("AUC:", metrics.roc_auc_score(y_test, np.sign(pred_test)))

    test_auc = metrics.roc_auc_score(y_test, np.sign(pred_test))
    return test_auc


if __name__ == '__main__':
    # 读取数据集
    x_train = np.loadtxt("adult_train_feature.txt")
    y_train = np.loadtxt("adult_train_label.txt")
    x_test = np.loadtxt("adult_test_feature.txt")
    y_test = np.loadtxt("adult_test_label.txt")

    AUC = []

    # 5折交叉AUC验证
    # auc_temp = adaboost(x_train, y_train, x_test, y_test, 31)
    # kf = KFold(n_splits=5, shuffle=True)
    #
    # for i in range(55):
    #     AUC_temp = 0.0
    #     for train, test in kf.split(x_train):
    #         AUC_temp += adaboost(x_train[train, :], y_train[train], x_train[test, :], y_train[test], i)
    #         print(i,":",AUC_temp)
    #     AUC.append(AUC_temp / 5.0)
    #     print("AUC:", AUC_temp/5.0)

    # 普通adaboost   AUC验证
    # for i in range(55):
    #     auc_temp = adaboost(x_train,y_train,x_test,y_test,i)
    #     AUC.append(auc_temp)
    #
    # plt.title("Adaboost ",fontsize = 24)
    # plt.xlabel("n_clf")
    # plt.ylabel('AUC')
    # plt.plot(range(55), AUC,c="green")
    # plt.show()
