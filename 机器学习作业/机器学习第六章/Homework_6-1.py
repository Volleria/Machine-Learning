from libsvm.svmutil import *

y, x = svm_read_problem('data.txt')
porblem = svm_problem(y, x)
# -s svm类型：SVM设置类型(默认0)
# -t 核函数类型：核函数设置类型(默认2)
# 　　0 – 线性：u’v
# 　　1 – 多项式：(ru’v + coef0)^degree
# 　　2 – RBF函数：exp(-gamma|u-v|^2)
# 　　3 –sigmoid：tanh(ru’v + coef0)
# -c cost：设置C-SVC，e-SVR和v-SVR的参数(损失函数)(默认1)
# -g r(gama)：核函数中的gamma函数设置(针对多项式/rbf/sigmoid核函数)(默认1/ k)

print("线性核:")
param1 = svm_parameter('-t 0 -c 6 -b 1')
model1 = svm_train(porblem, param1)
p_label1, p_acc1, p_val1 = svm_predict(y, x, model1)
print(p_label1)
print(p_acc1)
print(p_val1)


print("高斯核:")
param2 = svm_parameter('-t 2 -c 4 -b 1')
model2 = svm_train(porblem, param2)
p_label2, p_acc2, p_val2 = svm_predict(y, x, model2)
print(p_label2)
print(p_acc2)
print(p_val2)
