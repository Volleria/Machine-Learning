import math
from sklearn import tree
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from BoostMain import *
import random

# 读取数据集
x_train = np.loadtxt("adult_train_feature.txt")
y_train = np.loadtxt("adult_train_label.txt")
x_test = np.loadtxt("adult_test_feature.txt")
y_test = np.loadtxt("adult_test_label.txt")

# 获取样本个数以及属性个数
n_train = x_train.shape[0]
n_test = x_test.shape[0]
n_attribute = x_train.shape[1]

# 选取特征的数量
k = int(math.log(n_attribute, 2))


# 采样函数，同时对样本和属性进行采样
def sample(x_data, y_data, k):
    n_samples = x_data.shape[0]
    temp0 = np.zeros((x_data.shape[0], x_data.shape[1]))
    # 生产属性数大小的乱序数组
    temp = np.random.permutation(n_attribute)
    # 随机从中选取k个需要的属性
    attribute_need = np.random.choice(temp, size=k, replace=False)
    # 对训练集的属性进行随机采样
    # attribute_finished_train = x_train[:, attribute_need]
    for i in attribute_need:
        temp0[:, i] = x_data[:, i]
    # 随机选取的样本
    samples_need = np.random.choice(n_samples, size=int(n_samples), replace=True)
    # 对训练集的样本数进行采样
    samples_finished_train = temp0[samples_need, :]
    samples_finished_label = y_data[samples_need]
    # 返回采样后的数据集
    return samples_finished_train, samples_finished_label


def RF(x_train, y_train, x_test_my,y_test_my,M, k):
    # 初始化随机森林的预测结果
    y_pred_RF = np.zeros(int(x_test_my.shape[0]))

    for i in range(M):
        clf = tree.DecisionTreeClassifier(max_depth=2)
        x_train_sampled, y_train_sampled = sample(x_train, y_train, k)
        clf.fit(x_train_sampled, y_train_sampled)
        # 对每次预测结果×权重
        y_pred_train = clf.predict(x_test_my) / M
        # 加到最终结果上去
        y_pred_RF = y_pred_RF + y_pred_train

    #y_pred_RF  = [1 if x > 0.5 else 0 for x in y_pred_RF]
    test_auc = metrics.roc_auc_score(y_test_my,  y_pred_RF)
    # print("AUC:",test_auc)


    return test_auc


if __name__ == '__main__':
    AUC_RF = []
    AUC_Adaboost = []

    # RF(x_train,y_train,x_test,y_test,30,3)
# 普通RF
    for i in range(55):
        #k = random.randrange(6,8)
        auc_temp = RF(x_train,y_train,x_test,y_test,i,k)
        print("i  ", auc_temp)
        AUC_RF.append(auc_temp)

    plt.title("Random Forest ",fontsize = 24)
    plt.xlabel("n_clf")
    plt.ylabel('AUC')
    plt.plot(range(55), AUC_RF,c="blue")
    plt.show()
# 5折验证AUC
    # kf = KFold(n_splits=5)

    # for i in range(55):
    #     AUC_temp_RF = 0.0
    #     AUC_temp_Adaboost = 0.0
    #     for train, test in kf.split(x_train):
    #         AUC_temp_RF += RF(x_train[train, :], y_train[train], x_train[test, :], y_train[test],i + 1, k)
    #         AUC_temp_Adaboost += adaboost(x_train[train, :], y_train[train], x_train[test, :], y_train[test], i + 1)
    #
    #     AUC_RF.append(AUC_temp_RF / 5.0)
    #     AUC_Adaboost.append(AUC_temp_Adaboost / 5.0)
    #     print("i:", i, "AUC_RF:", AUC_temp_RF / 5.0)
    #     print("i:", i, "AUC_Ad:", AUC_temp_Adaboost / 5.0)
    #
    # plt.title("RESULT", fontsize=24)
    # plt.xlabel('n_clf')
    # plt.ylabel('AUC')
    # plt.plot((np.arange(55)), AUC_RF, color='green', label='Random Forest')
    # plt.plot((np.arange(55)), AUC_Adaboost, color='blue', label='Adaboost')
    # plt.show()
