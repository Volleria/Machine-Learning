from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

# 数据准备
x = np.loadtxt('Input.txt').reshape(-1, 1)
y = np.loadtxt('Output.txt')

# 自动选择合适的参数
svr = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0, C=0.5)
svr.fit(x, y)

gauss_svr = svr.fit(x, y)
y_pred = gauss_svr. predict(x)

plt.scatter(x, y, c='k', label='data', zorder=1)
plt.plot(x, y_pred, c='r', label='SVR_fit')
plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()
