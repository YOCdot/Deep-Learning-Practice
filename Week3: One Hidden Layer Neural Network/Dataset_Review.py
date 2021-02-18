import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
# testCases：提供了一些测试示例来评估函数的正确性
# planar_utils ：提供了在这个任务中使用的各种有用的功能


np.random.seed(1)  # 随机数种子

# 1、加载和查看数据集
X, Y = load_planar_dataset()
# X：一个numpy的矩阵，包含了这些数据点的数值
# Y：一个numpy的向量，对应着的是X的标签。红色:0，蓝色:1

plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)  # 绘制散点图
# 如果上面一条语句出现问题，使用下面语句：
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)  # 绘制散点图
# plt.show()
# 数据看起来像一朵红色（y=0）和一些蓝色（y=1）的数据点的花朵的图案。

# 目标是训练模型来拟合这些数据

shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]  # 训练集里样本的数量

print("数据信息：")
print("\tX的维度为：" + str(shape_X))
print("\tY的维度为：" + str(shape_Y))
print("\t数据集中样本的数量有" + str(m) + "个")


# 2、使用LogisticRegression的分类效果
# 使用神经网络前，先看看逻辑回归的表现如何。
# 可以使用sklearn的内置函数来做到这一点，运行下面的代码来训练数据集上的逻辑回归分类器。
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)  # 拟合
# 会报错，但无关紧要

# 画出LogisticRegression的分类
plot_decision_boundary(lambda x: clf.predict(x), X, Y)  # 绘制决策边界
plt.title("Logistic Regression")  # 图标题
LR_predictions = clf.predict(X.T)  # 预测结果
print("LogisticRegression的精确度： %d " % float((np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100)
      + "%" + "（正确标记的数据点所占的百分比）")
# plt.show()
# 准确性只有47%，因为数据集不是线性可分的，逻辑回归表现不佳。




