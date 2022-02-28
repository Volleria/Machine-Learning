from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 数据准备
x_train = np.genfromtxt('E:\Machine Learning\XIGUA\C5_NN\\train_feature.csv', delimiter=',')
y_train = np.genfromtxt('E:\Machine Learning\XIGUA\C5_NN\\train_target.csv', delimiter=',')
x_test = np.genfromtxt('E:\Machine Learning\XIGUA\C5_NN\\test_feature.csv', delimiter=',')


# 自动选择合适的参数
model = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0, C=0.5)
model.fit(x_train, y_train)

gauss_svr = model.fit(x_train, y_train)
y_pred = gauss_svr. predict(x_test)

y_predict = []
threshold = 0.5
for i in y_pred:
    if float(i) <= threshold:
        y_predict.append(0)
    else:
        y_predict.append(1)
test_target = pd.DataFrame(data=y_predict)
test_target.to_csv('test_target_h6.csv', index=False, encoding='gbk')
