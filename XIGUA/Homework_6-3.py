from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 数据准备
x_train = np.genfromtxt('E:\Machine Learning\XIGUA\C5_NN\\train_feature.csv', delimiter=',')
y_train = np.genfromtxt('E:\Machine Learning\XIGUA\C5_NN\\train_target.csv', delimiter=',')
x_test = np.genfromtxt('E:\Machine Learning\XIGUA\C5_NN\\test_feature.csv', delimiter=',')


# 使用高斯核训练
model1 = SVC(kernel='rbf', degree=3, gamma='auto', coef0=0, C=0.5)
model1.fit(x_train, y_train)

gauss_svc = model1.fit(x_train, y_train)
y_pred1 = gauss_svc. predict(x_test)

y_predict1 = []
threshold = 0.5
for i in y_pred1:
    if float(i) <= threshold:
        y_predict1.append(0)
    else:
        y_predict1.append(1)
test_target = pd.DataFrame(data=y_predict1)
print(y_predict1)
test_target.to_csv('test_target_rbf_SVM.csv', index=False, encoding='gbk')

# 使用线性核训练
model2 = SVC(kernel='linear', degree=3, gamma='auto', coef0=0, C=0.5)
model2.fit(x_train, y_train)

linear_svc = model2.fit(x_train, y_train)
y_pred2 = linear_svc. predict(x_test)

y_predict2 = []
threshold = 0.5
for i in y_pred2:
    if float(i) <= threshold:
        y_predict2.append(0)
    else:
        y_predict2.append(1)
test_target = pd.DataFrame(data=y_predict2)
print(y_predict2)
test_target.to_csv('test_target_linear_SVM.csv', index=False, encoding='gbk')