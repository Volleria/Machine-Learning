import numpy as np
import os
from PIL import Image
from sklearn.decomposition import PCA

# 维度
d = 20
# 文件路径
path = "D:\Machine Learning\XIGUA\机器学习_降维与度量学习\yalefaces"
# 保存image的list
X = []
# 原文件名，用于给输出图片命名
file_name = []

# 读取文件
for file in os.listdir(path):
    # 略过 readme.txt 文件
    if not file.endswith(".txt"):
        # 文件名切割
        var = file.title().split(".")[1]
        file_name.append(var)
        img = Image.open(os.path.join(path, file))
        # 将图片转换为矩阵
        img = np.array(img).reshape(img.width * img.height)
        X.append(img)


X = np.array(X)
# 调用sklearn中的PCA 方法
pca = PCA()
pca.fit(X)

for i in range(len(X)):
    x_hat = np.dot(pca.transform(X[i].reshape(1, -1))[:, :d], pca.components_[:d, :])
    x_hat += np.mean(X, axis=0)
    img_array = np.array(x_hat, dtype="int32").reshape(243, 320)
    img = Image.fromarray(img_array).convert("L")
    if i ==0:
        img.save(f"PCA_yalefaces/pca_{1}.{file_name[i]}.png")
    else:
        img.save(f"PCA_yalefaces/pca_{int((i-1)/11)+1}.{file_name[i]}.png")




