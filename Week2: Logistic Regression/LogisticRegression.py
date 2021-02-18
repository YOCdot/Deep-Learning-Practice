import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset


'''
建立神经网络的主要步骤是：
    1 定义模型结构（例如输入特征的数量）
    2 初始化模型的参数
    3 循环：
        3.1 计算当前损失（正向传播）
        3.2 计算当前梯度（反向传播）
        3.3 更新参数（梯度下降）
'''


# 数据预处理
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_y.shape[1]  # 训练集里图片的数量。
m_test = test_set_y.shape[1]  # 测试集里图片的数量。
num_px = train_set_x_orig.shape[1]  # 训练、测试集里面的图片的宽度和高度（均为64x64）。

# 现在看一看我们加载的东西的具体情况
print("训练集的数量: m_train = " + str(m_train))
print("测试集的数量 : m_test = " + str(m_test))
print("每张图片的宽/高 : num_px = " + str(num_px))
print("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("训练集_图片的维数 : " + str(train_set_x_orig.shape))
print("训练集_标签的维数 : " + str(train_set_y.shape))
print("测试集_图片的维数: " + str(test_set_x_orig.shape))
print("测试集_标签的维数: " + str(test_set_y.shape))

# 将训练集的维度降低并转置。
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# 将测试集的维度降低并转置。
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("训练集降维最后的维度： " + str(train_set_x_flatten.shape))
print("训练集_标签的维数 : " + str(train_set_y.shape))
print("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
print("测试集_标签的维数 : " + str(test_set_y.shape))

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


# 构建Sigmoid函数
def sigmoid(z):
    """
    Sigmoid函数
    :param z: 任何大小的标量或numpy数组
    :return s: Sigmoid(z)
    """

    s = 1 / (1 + np.exp(-z))

    return s


# 初始化参数w和b
def init_with_zeros(dim):
    """
    此函数为w创建一个维度为(dim, 1)的0向量，并将b初始化为0
    :param dim: 我们想要的w矢量的大小（或者这种情况下的参数数量）
    :return w: 维度为(dim, 1)的初始化向量
    :return b: 初始化的标量
    """

    w = np.zeros(shape=(dim, 1))
    b = 0
    # 使用断言来确保需要数据的正确性
    # 检查条件，不符合就终止程序，报错“AssertionError”
    assert (w.shape == (dim, 1))  # w的维度是(dim, 1)
    assert (isinstance(b, float) or isinstance(b, int))  # b的类型是float或int
    # isinstance()判断一个对象是否是一个已知的类型

    return w, b


def propagate(w, b, X, Y):
    """
    实现前向和后向传播的cost function和其梯度
    :param w: 权重，大小不等的数组(num_px * num_px * 3, 1)
    :param b: 偏差，一个标量
    :param X: 矩阵类型为(num_px * num_px * 3, 训练数量)
    :param Y: 真正的”标签“矢量（非猫为0，是猫为1），矩阵维度为(1, 训练数据数量)
    :return dw: 相对于w的损失梯度，形状与w相同
    :return db: 相对于b的损失梯度，形状与b相同
    :return cost: LogisticRegression的负对数似然成本
    """

    m = X.shape[1]

    # 正向传播
    A = sigmoid(np.dot(w.T, X) + b)  # 计算激活值（参考公式）
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # 计算本次正向传播的代价（参考公式）

    # 反向传播
    dw = (1 / m) * np.dot(X, (A - Y).T)  # 参考偏导dw计算公式
    db = (1 / m) * np.sum(A - Y)  # 参考偏导db计算公式

    # 使用断言保证数据正确性
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    # 创建一个字典，保存dw和db
    grads = {
        "dw": dw,
        "db": db
    }

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    此函数通过运行梯度下降算法来优化w和b
    提示：
    我们需要写下两个步骤并遍历它们：
        1）计算当前参数的成本和梯度，使用propagate()
        2）使用w和b的梯度下降法则更新参数
    :param w: 权重，大小不等的数组（num_px * num_px * 3，1）
    :param b: 偏差，一个标量
    :param X: 维度为（num_px * num_px * 3，训练数据的数量）的数组
    :param Y: 真正的”标签“矢量（非猫为0，是猫为1），矩阵维度为(1, 训练数据数量)
    :param num_iterations: 优化循环的迭代次数
    :param learning_rate: 学习率
    :param print_cost: 每100步打印一次损失值
    :return params: 包含权重w和偏差b的字典
    :return grads: 包含权重和偏差相对于成本函数的梯度的字典
    :return costs: 优化期间计算的所有成本列表，将用于绘制学习曲线
    """

    # 每次正向传播计算得到的代价记录入costs[]列表，得到所有的代价
    costs = []

    for i in range(num_iterations):

        # 获取梯度（字典）、每次正向传播得到的总体代价
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        # 梯度更新（梯度下降）
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录成本
        if i % 100 == 0:
            costs.append(cost)

        # 打印成本数据
        if print_cost and (i % 100 == 0):
            print("迭代次数： %i， 代价： %f" % (i, cost))

        params = {
            "w": w,
            "b": b
        }

        grads = {
            "dw": dw,
            "db": db
        }

    return params, grads, costs


def predict(w, b, X):
    """
    使用LogisticRegression参数w, b预测标签是0还是1
    :param w: 权重，大小不等的数组（num_px * num_px * 3，1）
    :param b: 偏差，一个标量
    :param X: 维度为（num_px * num_px * 3，训练数据的数量）的数据
    :return Y_prediction: 包含X中所有图片的所有预测{0,1}的一个numpy数组（向量）
    """

    m = X.shape[1]  # 图片的数量
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # 预测概率值
    A = sigmoid(np.dot(w.T, X) + b)  # y-hat = sigmoid(w.T * x + b)

    for i in range(A.shape[1]):
        # 将概率a[0,i]转换为实际预测p[0,i]
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0  # 预测值>0.5置1；预测值<0.5置0

    # 使用断言确定数据正确性
    assert (Y_prediction.shape == (1, m))

    return Y_prediction  # y帽


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    通过调用已实现的函数来构建LogisticRegression模型
    :param X_train: numpy的数组,维度为（num_px * num_px * 3，m_train）的训练集
    :param Y_train: numpy的数组,维度为（1，m_train）（矢量）的训练标签集
    :param X_test: numpy的数组,维度为（num_px * num_px * 3，m_test）的测试集
    :param Y_test: numpy的数组,维度为（1，m_test）的（向量）的测试标签集
    :param num_iterations: 表示用于优化参数的迭代次数的超参数
    :param learning_rate: 表示optimize（）更新规则中使用的学习速率的超参数
    :param print_cost: 设置为true以每100次迭代打印成本
    :return d: 包含有关模型信息的字典
    """

    # 初始化参数
    w, b = init_with_zeros(X_train.shape[0])

    # 计算参数（字典）、梯度（字典）、代价（列表）
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # 从参数（字典）中获取参数w和b
    w, b = parameters["w"], parameters["b"]
    # print("w:\n", w, "\nb:\n", b)

    # 对测试集、训练集实例进行预测（y帽）
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # 打印训练后的精度
    print("训练集精度：", format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100), "%")
    print("测试集精度：", format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100), "%")

    # 将结果存入模型信息字典
    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }
    # print(d)

    return d


# 这里加载的是真实的数据，请参见上面的代码部分
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# 画图
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations(per hundreds)')
plt.title('learning rate =' + str(d['learning_rate']))
plt.show()

# 比较几种不同的学习率取值
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("学习率：" + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i, print_cost=False)

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
