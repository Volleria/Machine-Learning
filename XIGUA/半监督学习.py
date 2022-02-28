import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn import svm
from sklearn.preprocessing import StandardScaler


def get_iris_data():
    iris = load_iris()
    # feature = iris['data']
    # label = iris['target']
    # 取iris数据集后100数据，即分类为2，3的数据集，构成一个二分类问题
    feature, label = iris.data[50:, :], iris.target[50:] * 2 - 3

    # 将样本属性标准化
    sc = StandardScaler()
    sc.fit(feature)
    feature = sc.transform(feature)

    # 30% 的测试样本
    test_feature = np.concatenate((feature[0:15], feature[50:65], feature[100:115]))
    test_label = np.concatenate((label[0:15], label[50:65], label[100:115]))

    # 10% 的有标记样本
    X1 = np.concatenate((feature[15:20], feature[65:70], feature[115:120]))
    Y1 = np.concatenate((label[15:20], label[65:70], label[115:120]))
    # 60% 的无标记样本
    X2 = np.concatenate((feature[20:50], feature[70:100], feature[120:150]))
    # 相加
    X3 = np.concatenate((X1, X2))
    return test_feature, test_label, X1, Y1, X2, X3


def get_wine_data():
    wine = load_wine()
    feature, label = wine.data[59:, :], wine.target[59:] * 2 - 3

    # 将样本属性标准化
    sc = StandardScaler()
    sc.fit(feature)
    feature = sc.transform(feature)

    # 30% 的测试样本
    test_feature = np.concatenate((feature[0:21], feature[71:85]))
    test_label = np.concatenate((label[0:21], label[71:85]))

    # 10% 的有标记样本
    X1 = np.concatenate((feature[21:28], feature[85:90]))
    Y1 = np.concatenate((label[21:28], label[85:90]))
    # 60% 的无标记样本
    X2 = np.concatenate((feature[28:71], feature[90:119]))
    # 相加
    X3 = np.concatenate((X1, X2))
    return test_feature, test_label, X1, Y1, X2, X3


def TSVM(test_feature, test_label, X1, Y1, X2, X3):
    # 调用sklearn中的SVM
    clf_svm = svm.SVC(C=1, kernel='linear')
    clf_svm.fit(X1, Y1)
    Y3_svm = clf_svm.predict(X3)

    clf_TSVM = svm.SVC(C=1, kernel='linear')
    clf_TSVM.fit(X1, Y1)

    # 用训练好的 SVM对Du进行预测
    Y2 = clf_TSVM.predict(X2)

    # 初始化cu,cl
    cu = 0.001
    cl = 1
    # 样本权重， 直接让有标签数据的权重为Cl,无标签数据的权重为Cu
    sample_weight = np.ones(len(X1) + len(X2))
    sample_weight[len(X1):] = cu
    # 初始化 id 数组
    id_set = np.arange(len(X2))

    while cu < cl:
        Y3 = np.concatenate((Y1, Y2))  # 合并有标签样本和无标签样本
        clf_TSVM.fit(X3, Y3, sample_weight=sample_weight)  # 对TSVM模型进行训练
        while True:
            Y2 = clf_TSVM.predict(X2)
            X2_dist = clf_TSVM.decision_function(X2)  # 计算无标签样本的距离
            norm_weight = np.linalg.norm(clf_TSVM.coef_)  # 进行标准化
            epsilon = 1 - X2_dist * Y2 * norm_weight

            plus_set, plus_id = epsilon[Y2 > 0], id_set[Y2 > 0]  # 正标记（1）样本
            minus_set, minus_id = epsilon[Y2 < 0], id_set[Y2 < 0]  # 负标记（-1）样本
            plus_max_id, minus_max_id = plus_id[np.argmax(plus_set)], minus_id[np.argmax(minus_set)]  # 找到最大、最小值的索引
            a, b = epsilon[plus_max_id], epsilon[minus_max_id]

            if a > 0 and b > 0 and a + b > 2:
                Y2[plus_max_id], Y2[minus_max_id] = -Y2[plus_max_id], -Y2[minus_max_id]  # 将无标签样本的预测值进行翻转
                Y3 = np.concatenate((Y1, Y2))  # 合并有标签样本和无标签样本的预测值
                clf_TSVM.fit(X3, Y3, sample_weight=sample_weight)  # 对TSVM模型进行训练
            else:
                break
        cu = min(cu * 2, cl)
        sample_weight[len(Y1):] = cu

    Y3 = np.concatenate((Y1, Y2))
    test_pred_svm = clf_svm.predict(test_feature)
    test_pred_TSVM = clf_TSVM.predict(test_feature)
    # print(test_label)
    # print("-----------------------")
    # print(test_pred_TSVM)
    score_svm = clf_svm.score(test_feature, test_label)  # SVM的模型精度
    score_TSVM = clf_TSVM.score(test_feature, test_label)  # TSVM的模型精度
    # fig = plt.figure(figsize=(4, 16))
    # ax = fig.add_subplot()
    ax = plt.subplot()
    ax.scatter(test_feature[:, 0], test_feature[:, 2], c=test_label, marker='o', cmap=plt.cm.coolwarm)
    plt.title('True Labels for test samples', fontsize=16)
    plt.show()

    ax1 = plt.subplot()
    ax1.scatter(test_feature[:, 0], test_feature[:, 2], c=test_pred_svm, marker='o', cmap=plt.cm.coolwarm)
    ax1.set_title('SVM, score: {0:.2f}%'.format(score_svm * 100), fontsize=16)
    plt.show()

    ax2 = plt.subplot()
    ax2.scatter(test_feature[:, 0], test_feature[:, 2], c=test_pred_TSVM, marker='o', cmap=plt.cm.coolwarm)
    ax2.set_title('TSVM, score: {0:.2f}%'.format(score_TSVM * 100), fontsize=16)
    plt.show()

    return score_svm, score_TSVM


if __name__ == '__main__':
    # test_feature, test_label, X1, Y1, X2, X3 = get_iris_data()
    test_feature, test_label, X1, Y1, X2, X3 = get_wine_data()
    score_svm, score_TSVM = TSVM(test_feature, test_label, X1, Y1, X2, X3)
    print("iris_dataste:")
    print("SVM的模型精度：", score_svm)
    print("TSVM的模型精度：", score_TSVM)
