import matplotlib.pyplot as plt
import csv

# 初始化各个变量
label = []
recall_list = []
precision_list = []
FPR = []
TP = 0.0
FP = 0.0

# 读取csv文件并将其中的数据保存在对应的list中
f = csv.reader(open('data.csv', 'r'))
l = []
for i in f:
    label.append(i[1])
    l.append([i[1], i[2]])

# 删去不必要的表头信息
del label[0], l[0]

# 计算出真实情况中的正例和反例数量
M = label.count('1')
N = label.count('0')

# 对l中的内容进行第二关键字的排序  降序
l.sort(key=lambda x: float(x[1]), reverse=True)

# 对list进行遍历，从中计算出TP,FP的值，此处用了双重循环的，可以采用dp化简
for i in range(500):
    for k in range(i):
        if float(l[k][0]) == 1.0:
            TP += 1.0
        else:
            FP += 1.0
    # 分母不能为0
    if (TP + FP) != 0.0:
        precision = TP / (TP + FP)
        recall = TP / M
        fpr = FP / N
        precision_list.append(precision)
        recall_list.append(recall)
        FPR.append(fpr)
    TP = 0.0
    FP = 0.0


# 采用PPT上的方法 计算 AUC 的值
AUC = 0.0
for i in range(498):
    AUC = AUC + ((FPR[i + 1] - FPR[i]) * (recall_list[i] + recall_list[i + 1]))
AUC = 0.5 * AUC

# 输出P-R图 和 ROC图
plt.plot(recall_list, precision_list)
plt.title("P-R")
plt.ylabel("precision")
plt.xlabel("recall")
plt.xlim([-0.001, 1.01])
plt.ylim([-0.001, 1.01])
plt.savefig("P-R.png")
plt.show()

plt.plot(FPR, recall_list)
plt.title("ROC")
plt.ylabel("TPR")
plt.xlabel("FPR")
plt.xlim([-0.001, 1.01])
plt.ylim([-0.001, 1.01])
plt.savefig("ROC.png")
plt.show()
