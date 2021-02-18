import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# testCases：提供了一些测试示例来评估函数的正确性
# planar_utils ：提供了在这个任务中使用的各种有用的功能


"""
    构建神经网络的一般方法：

        1、定义神经网络结构（输入单元的数量，隐藏单元的数量等）。
        2、初始化模型的参数
        3、循环：
            3.1、实施前向传播
            3.2、计算损失
            3.3、实现向后传播
            3.4、更新参数（梯度下降）
"""

# 把以上功能合并到一个函数nn_model()中，当构建好nn_model()并学习了正确的参数，就可以使用它来预测新的数据。


"""
    一、定义神经网络结构
    
        在构建之前，我们要先把神经网络的结构给定义好：
            n_x: 输入层的数量
            n_h: 隐藏层的数量（这里设置为4）
            n_y: 输出层的数量
"""


def layer_sizes(X, Y):
    """
    获得数据集信息并由此信息构造神经网络总体架构

    :param X: 输入，其维度：（输入的数量， 训练/测试的数量）
    :param Y: 标签，其维度：（输出的数量， 训练/测试的数量）

    :return n_x: 输入层神经元个数
    :return n_h: 隐层神经元个数
    :return n_y: 输出层神经元个数
    """

    n_x = X.shape[0]  # 输入层
    n_h = 4  # 隐层，硬编码4个神经元
    n_y = Y.shape[0]  # 输出层

    return n_x, n_h, n_y


# # 测试layer_sizes()
# print("=========================测试layer_sizes=========================")
# X_asses, Y_asses = layer_sizes_test_case()  # 获得一个(5, 3)维和(2, 3)的高斯分布随机数组
# (n_x, n_h, n_y) = layer_sizes(X_asses, Y_asses)  # 获得神经网络总体架构
# print("输入层: n_x = ", str(n_x), "个神经元")
# print("隐层: n_h = ", str(n_h), "个神经元")
# print("输出层: n_y = ", str(n_y), "个神经元")


"""
    二、初始化模型参数
    
        实现函数initialize_parameters()。
        为确保我们的参数大小合适，请参考上面的神经网络结构。
        使用随机值初始化权重矩阵：
            np.random.randn(a，b)* 0.01来随机初始化一个维度为(a，b)的矩阵。
        将偏向量初始化为零：
            np.zeros((a，b))用零初始化矩阵（a，b）。
"""


def initialize_parameters(n_x, n_h, n_y):
    """
    初始化神经网络参数。W不能为0；b为0。

    :param n_x: 输入层神经元个数
    :param n_h: 隐层神经元个数
    :param n_y: 输出层神经元个数

    :return parameters: 包含参数的字典
        W1: 输入层-->隐层间的权重矩阵,维度为(n_h，n_x)
        b1: 输入层-->隐层间偏置向量，维度为(n_h，1)
        W2: 隐层-->输出间权重矩阵，维度为(n_y，n_h)
        b2: 隐层-->输出间偏置向量，维度为(n_y，1)
    """

    np.random.seed(2)

    # 初始化
    W1 = np.random.randn(n_h, n_x) * 0.01  # 后面乘一个小实数
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01  # 后面乘一个小实数
    b2 = np.zeros(shape=(n_y, 1))

    # 使用断言保证数据正确性
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    # 参数矩阵合并为字典作为输出
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# # 测试initialize_parameters
# print("=========================测试initialize_parameters=========================")
# n_x, n_h, n_y = initialize_parameters_test_case()
# parameters = initialize_parameters(n_x, n_h, n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


"""
    三、迭代循环
        迭代循环中有三个环节：
        1、前向传播
        2、cost计算
        3、反向传播
"""

"""
    迭代环节一
    
    1、前向传播
        实现前向传播函数forward_propagation()
        激活函数可以选择sigmoid()或np.tanh()
        
    步骤如下：
        1、使用字典类型的parameters（它是initialize_parameters() 的输出）检索每个参数。
        2、实现向前传播,计算Z^[1]、A^[1]、Z^[2]、A^[2]（训练集里面所有例子的预测向量）。
        3、反向传播所需的值存储在“cache”中，cache将作为反向传播函数的输入。
"""


def forward_propagation(X, parameters):
    """
    接收参数并进行前向传播计算函数。

    :param X: 输入数据，维度为(n_x, m)
    :param parameters: 初始化函数initialize_parameters()输出的参数字典

    :return A2: sigmoid(Z2)，第二层的激活值
    :return cache: 包含Z^[1]、A^[1]、Z^[2]、A^[2]的字典类型变量
    """

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # 前向传播计算A^[2]
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    # 使用断言确保数据正确性
    assert (A2.shape == (1, X.shape[1]))

    # 计算结果矩阵合并为字典并输出
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


# # 测试forward_propagation
# print("=========================测试forward_propagation=========================")
# X_assess, parameters = forward_propagation_test_case()
# A2, cache = forward_propagation(X_assess, parameters)
# print(np.mean(cache["Z1"]), np.mean(cache["A1"]), np.mean(cache["Z2"]), np.mean(cache["A2"]))


"""
    迭代环节二
    
    2、cost计算
        交叉熵损失计算公式。笔记中有，J(w, b)
        
    计算方法
        有很多方法可以计算交叉熵损失，在Python中可以如此实现:
        logprobs = np.multiply(np.log(A2),Y)
        cost = - np.sum(logprobs)  # 不需要使用循环就可以直接算出来。
        当然，也可以使用np.multiply()然后使用np.sum()或者直接使用np.dot()
"""


def compute_cost(A2, Y, parameters):
    """
    计算交叉熵成本方程。

    :param A2: sigmoid(Z2)，第二层的激活值
    :param Y: "True"标签向量，维度为（1, 样本个数）
    :param parameters: 参数字典

    :return cost: 交叉熵成本
    """

    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # cost计算
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m
    cost = float(np.squeeze(cost))

    assert (isinstance(cost, float))

    return cost


# # 测试compute_cost
# print("=========================测试compute_cost=========================")
# A2, Y_assess, parameters = compute_cost_test_case()
# print("cost = " + str(compute_cost(A2, Y_assess, parameters)))


"""
    迭代环节三
    
    3、反向传播
        步骤一、反向传播
            实现反向传播函数backward_propagation()（吴老师总结6个方程）
        
    计算方式
        使用正向传播期间计算的cache实现反向传播。
        为计算dZ1，需计算g[1]'(Z[1])，其中g[1]()是激活函数。
        因为da[1] = g[1](z)，那么g[1]'(z) = 1 - a^2（对sigmoid求导得到）
        使用( 1 - np.power(A1, 2) )来计算g[1]'(Z[1])
"""


def backward_propagation(parameters, cache, X, Y):
    """
    使用上述说明搭建反向传播函数。

    :param parameters: 参数字典
    :param cache: 包含Z^[1]、A^[1]、Z^[2]、A^[2]的字典类型变量
    :param X: 输入数据，维度为(2, 样本个数)
    :param Y: “True”标签，维度为(1, 样本个数)

    :return grads: 梯度字典，包含dW1、db1、dW2、db2
    """

    # 获取样本个数
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


# # 测试backward_propagation()
# print("=========================测试backward_propagation=========================")
# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
# print("dW1 = " + str(grads["dW1"]))
# print("db1 = " + str(grads["db1"]))
# print("dW2 = " + str(grads["dW2"]))
# print("db2 = " + str(grads["db2"]))


"""
    迭代环节三
    
    3、反向传播
        步骤二、参数更新
            使用(dW1, db1, dW2, db2)来更新(W1, b1, W2, b2)。
            
    计算方式
        θ = θ - α(dJ / dθ)，偏导
        α 学习率
        θ 参数
"""


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    使用上述梯度下降公式更新参数。

    :param parameters: 参数字典
    :param grads: 梯度字典，包含dW1、db1、dW2、db2
    :param learning_rate: 学习率

    :return parameters: 更新后的参数字典
    """

    W1, W2 = parameters["W1"], parameters["W2"]
    b1, b2 = parameters["b1"], parameters["b2"]

    dW1, dW2 = grads["dW1"], grads["dW2"]
    db1, db2 = grads["db1"], grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# # 测试update_parameters
# print("=========================测试update_parameters=========================")
# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


"""
    迭代环节三

    3、反向传播
        步骤三、功能整合
            将先前实现的功能整合入函数nn_model()中，神经网络模型必须以正确的顺序使用先前实现的功能。
"""


def nn_model(X, Y, n_h, num_iterations, print_cost=False):
    """
    整合了反向传播计算和参数更新的函数。

    :param X: 输入，维度为(2, 样本数)
    :param Y: 标签，维度为(1, 样本数)
    :param n_h: 隐层神经元个数
    :param num_iterations: 迭代次数
    :param print_cost: 默认为False。若为True，每1000次迭代打印一次cost计算数值

    :return parameters: 模型学得的参数，可用来进行预测
    """

    np.random.seed(3)

    # 获输入数据和标签数据
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    # 由此信息构造神经网络总体架构

    # 初始化参数
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # 参数迭代更新
    for i in range(num_iterations):

        # 获得第二层的激活值A2、包含Z^[1]、A^[1]、Z^[2]、A^[2]的字典cache
        A2, cache = forward_propagation(X, parameters)
        # 获得cost
        cost = compute_cost(A2, Y, parameters)
        # 获得梯度字典
        grads = backward_propagation(parameters, cache, X, Y)
        # 更新参数，指定学习率为1.2
        parameters = update_parameters(parameters, grads, learning_rate=1.2)

        # 是否打印cost
        if print_cost:
            if i % 1000 == 0:
                print("第", i, "次循环，cost为：", str(cost))

    # 返回迭代更新后的参数字典
    return parameters


# # 测试nn_model()
# print("=========================测试nn_model=========================")
# X_assess, Y_assess = nn_model_test_case()
#
# parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
"""
    两个警告：
        问题应该是w的值不合适造成的（太大或太小），造成了z值太大，在计算sigmoid函数时e^x容易向上溢出。
        
        1、RuntimeWarning: divide by zero encountered in log警告:
            第219行的问题
            代码：logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
            log(x)因为输入的x值太大（正值）或太小（负值），产生了内存溢出。
            所以在cost函数中的log计算引发此警告。
        
        2、RuntimeWarning: overflow encountered in exp
            planar_utils.py中第26行的问题（计算sigmoid的过程）
            代码：s = 1 / (1 + np.exp(-x))
            计算激活值时，使用exp(-x)计算e^(-x)，参考e^(-x)函数图，输入的z值太小（负），e^(-x)会变得很大，容易向上溢出。
"""


"""
    四、使用神经网络进行预测
    
        构建函数predict()来使用模型进行预测， 使用向前传播来预测结果。
        二分类器，判别方式：
            激活值 > 0.5：y^ = 1
            激活值 < 0.5：y^ = 0
"""


def predict(parameters, X):
    """
    使用学得的参数对X中的实例进行预测
    :param parameters: 参数字典
    :param X: 输入数据，维度为(n_x, m)
    :return predictions: 预测的向量（0：红色， 1：蓝色）
    """

    # 前向传播
    A2, cache = forward_propagation(X, parameters)
    # 获得预测向量，np.round取四舍五入后
    predictions = np.round(A2)

    # 返回预测值
    return predictions


# # 测试predict
# print("=========================测试predict=========================")
#
# parameters, X_assess = predict_test_case()
#
# predictions = predict(parameters, X_assess)
# print("预测的平均值 = " + str(np.mean(predictions)))


"""
    正式运行
        在数据集上运行该单隐层神经网络
"""

if __name__ == "__main__":

    # 加载数据集
    X, Y = load_planar_dataset()
    # X：一个numpy的矩阵，包含了这些数据点的数值
    # Y：一个numpy的向量，对应着的是X的标签。红色:0，蓝色:1

    # 获得参数
    parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

    # 绘制边界
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))

    # 进行预测并计算准确率
    predictions = predict(parameters, X)
    print('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

    # 显示图表
    plt.show()
