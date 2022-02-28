import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)

Train_data = np.array([
    [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
    [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
    [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
    [0.593, 0.042], [0.719, 0.103]])

Train_Size = Train_data.shape[0]
Train_label = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
tree.fit(Train_data, Train_label)


def plot_decision_boundary(clf, axes):
    xp = np.linspace(axes[0], axes[1], 100)
    yp = np.linspace(axes[2], axes[3], 100)
    x1, y1 = np.meshgrid(xp, yp)
    xy = np.c_[x1.ravel(), y1.ravel()]
    y_pred = clf.predict(xy).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, y1, y_pred, alpha=0.3, cmap=custom_cmap)


y_pred = tree.predict(Train_data)
print(y_pred)
plot_decision_boundary(tree, axes=[0, 1, 0, 1])
p1 = plt.scatter(Train_data[y_pred == 0, 0], Train_data[y_pred == 0, 1], color='blue',marker="o")
p2 = plt.scatter(Train_data[y_pred == 1, 0], Train_data[y_pred == 1, 1], color='green',marker="o")
# 设置注释
plt.legend([p1,p2],["Bad Melon","Good Melon"],loc='upper right')
plt.show()
