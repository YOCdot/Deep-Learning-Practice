import numpy as np
import h5py
import matplotlib.pyplot as plt

import testCases
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import lr_utils


np.random.seed(1)


"""
    深层神经网络

    多层的神经网络的结构是：
        输入层->隐藏层->隐藏层->···->隐藏层->输出层**
        每一层，首先计算Z = np.dot(W,A) + b，这叫做【linear_forward】（线性计算部分）
        然后再计算 A = relu(Z) 或者 A = sigmoid(Z)，这叫做【linear_activation_forward】（线性计算+激活计算部分）
        合并起来就是这一层的计算方法。
        所以每一层的计算都有两个步骤:先计算Z，再计算A。
        （通常讲两个步骤合并，直接计算A，也就是【linear_activation_forward】）
        
    构建深层神经网络的步骤：
        1、初始化网络参数
        2、前向传播
            2.1、计算一层中的线性求和的部分
            2.2、计算激活函数的部分（ReLU使用L-1次，Sigmoid使用1次）
            2.3、结合线性求和与激活函数
        3、计算误差
        4、反向传播
            4.1、线性部分的反向传播公式
            4.2、激活函数部分的反向传播公式
            4.3、结合线性部分与激活函数的反向传播公式
        5、更新参数

    注意，对于每个前向函数，都有一个相应的后向函数。
    所以通常每一步计算都会向cache中存储一些值，cache中存储的值对反向传播时的梯度计算很有用，在反向传播模块中，我们将使用cache来计算梯度。
    
    此次编程分别构建
        两层神经网络、多层神经网络。
"""


"""
    一、初始化参数
        对于一个两层的神经网络而言，其模型结构是：线性计算 --> ReLU --> 线性计算 --> sigmoid
"""


# 参数初始化函数（2层）
def initialize_parameters(n_x, n_h, n_y):
    """
    此函数是为了初始化二层网络参数而使用的函数
    :param n_x: 输入层单元数
    :param n_h: 隐层单元数
    :param n_y: 输出层单元数
    :return parameters: 参数字典
        W1 - 权重矩阵（输入层 --> 隐层），维度(n_h, n_x)
        b1 - 偏置向量（输入层 --> 隐层），维度(n_h, 1)
        W2 - 权重矩阵（隐层 --> 输出层），维度(n_y, n_h)
        b2 - 偏置向量（隐层 --> 输出层），维度(n_y, 1)
        （维度：(n[l], n[l-1])）
    """

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    # 保持数据正确性
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    # 参数字典
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    # 输出参数字典
    return parameters


# # 测试initialize_parameters()
# print("==============测试initialize_parameters==============")
# parameters = initialize_parameters(3, 2, 1)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


# 参数初始化函数（多层模型）
def initialize_parameters_deep(layers_dims):
    """
    此函数是为了初始化深层网络的参数而使用的函数。

    :param layers_dims: 包含改深层网络中每层单元数量的列表

    :return:
        parameters:参数字典（W1、b1...WL、bL）{
            Wl - 权重矩阵，维度(layers_dims[l], layers_dims[l - 1])
            bl - 偏置向量，维度(layers_dims[l], 1)}
    """

    # 确定随机数种子
    np.random.seed(3)

    parameters = {}

    # 获取层数L
    L = len(layers_dims)

    # 从第一层迭代到最后一层
    for l in range(1, L):

        # "Wl": (第l层单元数量, 第l-1层单元数量) / sqrt(第l-1层单元数量)
        # parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01  # 普通初始化
        # Xavier初始化：思想就是尽可能的让输入和输出服从相同的分布，即保持输入和输出的方差一致。
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(layers_dims[l - 1])

        # "bl": (第l层单元数量, 1)
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        # 确保数据正确性
        assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters


# # 测试initialize_parameters_deep()
# print("==============测试initialize_parameters_deep==============")
# layers_dims = [5, 4, 3]
# parameters = initialize_parameters_deep(layers_dims)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


"""
    二、前向传播
        
        前向传播有以下三个步骤
        
            2.1、计算一层中的线性求和的部分
                LINEAR
                
            2.2、计算激活函数的部分（ReLU使用L-1次，Sigmod使用1次）
                LINEAR - >ACTIVATION，其中激活函数将会使用ReLU或Sigmoid。
                
            2.3、结合线性求和与激活函数
                [LINEAR - > RELU] ×（L-1） - > LINEAR - > SIGMOID（整个模型）
        
        (使用向量化后的公式Z[l] = W[l]A[l-1]+b[l]进行计算)
"""


"""
    2.1、计算一层中的线性求和的部分
        【LINEAR】线性部分
        
    (使用向量化后的公式Z[l] = W[l]A[l-1]+b[l]进行计算)
"""


def linear_forward(A, W, b):
    """
    实现前向传播的线性部分。

    :param A: 上一层（或输入数据）激活后的值，维度(l-1层单元数, 样本数)
    :param W: 权重矩阵，numpy数组，维度(l层单元数, l-1层单元数)
    :param b: 偏置向量，numpy向量，维度(l层单元数, 1)

    :return:
        Z: 激活函数的输入，也称为预激活参数
        cache: 一个包含“A”，“W”和“b”的字典，存储这些变量以有效地计算后向传递
    """

    # 线性值计算
    Z = np.dot(W, A) + b

    # 确保数据正确性
    assert (Z.shape == (W.shape[0], A.shape[1]))

    # 缓存
    cache = (A, W, b)

    return Z, cache


# # 测试linear_forward()
# print("==============测试linear_forward==============")
# A, W, b = testCases.linear_forward_test_case()
# Z, linear_cache = linear_forward(A, W, b)
# print("Z = " + str(Z))


"""
    2.2、计算激活函数的部分（ReLU使用L-1次，Sigmod使用1次）
        【ACTIVATION】线性激活部分.
        激活函数将会使用ReLU（前L-1层）或Sigmoid（第L层）。
        
       

    2.3、结合线性求和与激活函数
        [LINEAR -> RELU] ×（L-1） - > LINEAR - > SIGMOID（整个模型）
        
         使用公式：
            A[l] = g[l](Z[l]) = g[l](W[l]A[l] + b[l])
            
        这里把两个功能（线性和激活）合并为一个功能【LINEAR-> ACTIVATION】。
        因此，需实现一个函数先执行【LINEAR】部分，然后执行【ACTIVATION】部分。
"""


def linear_activation_forward(A_prev, W, b, activation):
    """
    实现【LINEAR -> ACTIVATION】线性激活部分。

    :param A_prev: 上一层（或输入层）激活后的值，维度(l-1层单元数, 样本数)
    :param W: 权重矩阵，numpy数组，维度为(l层单元数, l-1层单元数)
    :param b: 偏置向量，numpy向量，维度为(l层单元数, 1)
    :param activation: 本层使用的激活函数，字符串类型，{"sigmoid" | "relu"}

    :returns:
        A: 激活函数的输出，也称激活后的值
        cache: 包含“linear_cache”和“activation_cache”的字典，我们需要存储它以有效地计算后向传递
    """

    # 激活函数为sigmoid
    if activation == "sigmoid":

        # 本层的Z, linear_cache(上层的A, 本层的W, 本层的b)
        Z, linear_cache = linear_forward(A_prev, W, b)
        # 本层的A(Sigmoid), activation_cache(本层的Z)
        A, activation_cache = sigmoid(Z)

    # 激活函数为relu
    elif activation == "relu":

        # 本层的Z, linear_cache(上层的A, 本层的W, 本层的b)
        Z, linear_cache = linear_forward(A_prev, W, b)
        # 本层的A(ReLU), activation_cache(本层的Z)
        A, activation_cache = relu(Z)

    # 确保数据正确性
    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    # 缓存cache(上层的A, 本层的W, 本层的b, 本层的Z)由两部分组成:
    # linear_cache(上层的A, 本层的W, 本层的b)
    # activation_cache(本层的Z)
    cache = (linear_cache, activation_cache)

    return A, cache


# # 测试linear_activation_forward()
# print("==============测试linear_activation_forward==============")
# A_prev, W, b = testCases.linear_activation_forward_test_case()
# # Sigmoid
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="sigmoid")
# print("Sigmoid，A = " + str(A))
# # ReLU
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="relu")
# print("ReLU，A = " + str(A))


"""
    上面已经实现了两层模型需要的前向传播函数，那多层网络模型的前向传播是怎样的呢？
    实现多层神经网络模型的前向传播需要调用上面两个函数。
    为了更方便地实现L层神经网络，需要一个函数来复制前一个函数（带有RELU的linear_activation_forward）L-1次，
    然后用一个带有SIGMOID的linear_activation_forward跟踪它。
    
    在下面的代码中，AL表示A[L] = σ(Z[L]) = σ(W[L]A[L-1] + b[L]), 也就是Y^
"""


def L_model_forward(X, parameters):
    """
    实现多层神经网络的前向传播。
    :param X: 输入数据，numpy数组，维度(输入单元数，样本数)
    :param parameters: 参数初始化函数（多层模型）initialize_parameters_deep()的输出
    :return:
        AL: 最后的激活值
        caches: 包含以下内容的缓存列表
            linear_relu_forward()的每个cache(共L-1个，索引从0 --> L-2)
            linear_sigmoid_forward()的cache(仅一个，索引L-1)
    """

    caches = []

    # 输入X，并将其赋值给A[0]，用于计算g[1](Z[1])
    A = X

    # 获取深层神经网络的层数L（整除）
    L = len(parameters) // 2

    # 在1到L-1层进行迭代
    for l in range(1, L):  # i from 1 to L-1

        # 第l层【线性+激活】计算，激活函数使用ReLU
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)

    # 第L层【线性+激活】计算，激活函数使用Sigmoid
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    # 检查数据正确性
    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


# # 测试L_model_forward()
# print("==============测试L_model_forward==============")
# X, parameters = testCases.L_model_forward_test_case()
# AL, caches = L_model_forward(X, parameters)
# print("AL = " + str(AL))
# print("caches 的长度为 = " + str(len(caches)))


"""
    三、计算成本
    
        前向传播计算结束后需要进行cost计算，以确定模型是否进步。（与标签进行比对并计算）
            cost计算公式：
            - 1/m ∑ i=1->m ( y^(i)log(a^[L](i)) + (1 - y^(i))log(1 - a^[L](i) ))
"""


def compute_cost(AL, Y):
    """
    使用上述公式计算成本函数。

    :param AL: 与标签预测相对应的概率向量（就是Y^），维度(1, 样本数)
    :param Y: 标签向量（0-非猫，1—猫，），维度(1, 样本数)

    :return:
        cost: 交叉熵成本
    """

    # 获取样本数
    m = Y.shape[1]

    # 交叉熵计算公式
    cost = - np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m

    # 降维，便于输出，去掉中括号[]
    cost = np.squeeze(cost)

    # 检查数据正确性
    assert (cost.shape == ())

    return cost


# # 测试compute_cost()
# print("==============测试compute_cost==============")
# Y, AL = testCases.compute_cost_test_case()
# print("cost = " + str(compute_cost(AL, Y)))


"""
    四、反向传播
        反向传播用于计算相对于参数的损失函数的梯度。
        
        此过程需要使用dZ^[l]来计算三个输出dW^[l], db^[l], dA^[l]，以下是求取该三个输出的公式：
            dW^[l] = dL / dW^[l] = 1/m * dZ^[l] * A^[l-1]T
            db^[l] = dL / db^[l] = 1/m * ∑ i=1->m dZ^[l](i)
            dA^[l-1] = dL / dA^[l-1] = W^[l]T * dZ^[l]
            
        和前向传播类似，我们有需要使用三个步骤来构建反向传播：
            1、【LINEAR】线性部分反向计算
            2、【LINEAR -> ACTIVATION】线性+激活部分反向计算，其中【ACTIVATION】计算Relu或者Sigmoid 的结果
            3、【[LINEAR -> RELU] × (L-1) -> LINEAR -> SIGMOID】反向计算(整个模型)
"""


# 1、【LINEAR backward】线性部分反向计算（激活函数内套部分Z）
def linear_backward(dZ, cache):
    """
    为单层（第l层）实现反向传播的线性部分（这里默认dZ已经求得并作为参数传入，函数求J对W, b, A^[l-1]的导）
    :param dZ: 相对于（当前第l层的）线性输出的成本梯度
    :param cache: 来自当前层前向传播的值的元组(A_prev，W，b)
    :return:
        dA_prev: 相对于激活（前一层l-1）的成本梯度，与A_prev维度相同
        dW: 相对于W（当前层l）的成本梯度，与W的维度相同
        db: 相对于b（当前层l）的成本梯度，与b维度相同
    """

    # 获取当前层的正向传播的cache
    A_prev, W, b = cache

    # 获取样本数(单元个数, 样本数)
    m = A_prev.shape[1]

    # 梯度计算（笔记中有公式，向量化版本）
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    # 计算得到dA^[l-1]
    dA_prev = np.dot(W.T, dZ)

    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev, dW, db


# # 测试linear_backward()
# print("==============测试linear_backward==============")
# dZ, linear_cache = testCases.linear_backward_test_case()
#
# dA_prev, dW, db = linear_backward(dZ, linear_cache)
# print("dA_prev = " + str(dA_prev))
# print("dW = " + str(dW))
# print("db = " + str(db))


"""
    2、【LINEAR -> ACTIVATION】线性+激活部分反向计算
        dnn_utils.py提供了两个后向函数:
            2.1、sigmoid_backward:实现了sigmoid()函数的反向传播，你可以这样调用它：
                dZ = sigmoid_backward(dA, activation_cache)
            2.2、relu_backward: 实现了relu()函数的反向传播，你可以这样调用它：
                dZ = relu_backward(dA, activation_cache)
        
        如果g(.)是激活函数, 那么sigmoid_backward()和relu_backward()这样计算：
            dZ^[l] = dA^[l] * g'(Z^[l])
"""


def linear_activation_backward(dA, cache, activation="relu"):
    """
    实现【LINEAR -> ACTIVATION】线性+激活部分反向计算。
    :param dA: 当前层l的激活后的梯度值
    :param cache: 我们存储的用于有效计算反向传播的值的元组（值为linear_cache，activation_cache）
    :param activation: 要在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】
    :return:
        dA_prev - 相对于激活（前一层l-1）的成本梯度值，与A_prev维度相同
        dW - 相对于W（当前层l）的成本梯度值，与W的维度相同
        db - 相对于b（当前层l）的成本梯度值，与b的维度相同
    """

    # linear_activation_forward()的返回值为：
    # cache = (linear_cache, activation_cache)
    # linear_cache(上层的A, 本层的W, 本层的b)
    # activation_cache(本层的Z)
    linear_cache, activation_cache = cache

    # 当激活函数选择ReLU时
    if activation == "relu":

        # 使用函数relu_backward()计算反向传播dZ（外套，激活函数求导）
        dZ = relu_backward(dA, activation_cache)
        # 使用函数linear_backward()计算反向传播dA^[l-1]、dW^[l]、db^[l]（内含，线性部分求导）
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    # 当激活函数选择Sigmoid时
    elif activation == "sigmoid":

        # 使用函数sigmoid_backward()计算激活函数反向传播dZ（外套，激活函数求导）
        dZ = sigmoid_backward(dA, activation_cache)
        # 使用函数linear_backward()计算线性部分反向传播dA^[l-1]、dW^[l]、db^[l]（内含，线性部分求导）
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


# # 测试linear_activation_backward()
# print("==============测试linear_activation_backward==============")
# AL, linear_activation_cache = testCases.linear_activation_backward_test_case()
#
# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation="sigmoid")
# print("sigmoid:")
# print("dA_prev = " + str(dA_prev))
# print("dW = " + str(dW))
# print("db = " + str(db) + "\n")
#
# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation="relu")
# print("relu:")
# print("dA_prev = " + str(dA_prev))
# print("dW = " + str(dW))
# print("db = " + str(db))


"""
    3、【[LINEAR -> RELU] × (L-1) -> LINEAR -> SIGMOID】反向计算(整个模型)
        在前向计算中，我们存储了一些包含包含（X，W，b和z）的cache，反向传播需要使用它们来计算梯度值。
        在L层模型中，我们需要从第L层反向遍历所有的隐层，每一步都需要使用本层的cache值来进行反向传播。
        
        先前计算了A^[L]，就是y^，是输出层的输出，A^[L] = σ(Z^[L])
        反向传播从A^[L]开始，所以需要计算dAL，公式：
            dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        得到dAL后，可以使用此激活后的梯度dAL继续反向计算。
        
        构建多层模型反向传播函数。
"""


def L_model_backward(AL, Y, caches):
    """

    :param AL: 概率向量，正向传播的输出（L_model_forward()的返回值）
    :param Y: 标签向量（0-非猫，1—猫，），维度(1, 样本数)
    :param caches: 包含以下内容的cache列表(下标从0开始)：
                 linear_activation_forward（"relu"）的cache，前L-1层（不包含输出层）
                 linear_activation_forward（"sigmoid"）的cache，第L层（输出层）
    :return:
        grads: 梯度字典{
            grads [“dA”+ str（l）] = ...
            grads [“dW”+ str（l）] = ...
            grads [“db”+ str（l）] = ...}
    """

    grads = {}

    # 获取层数L
    L = len(caches)

    # 获取样本数m，AL是输出层的输出，维度(1, 样本数)
    m = AL.shape[1]

    # 获取标签向量Y，并将其维度与输出向量AL对齐
    Y = Y.reshape(AL.shape)

    # 只有最后一层使用了sigmoid，所以使用公式计算出最后一层的dA^[L]
    # 计算损失函数Loss(A, Y)对输出激活值A^[L]的梯度（笔记公式）：-(y/a + (1-y)/(1-a))
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # caches = (linear_cache, activation_cache) = (上层的A, 本层的W, 本层的b, 本层的Z)
    # linear_cache(上层的A, 本层的W, 本层的b) + activation_cache(本层的Z)
    # 下标从0开始，所以要L-1

    # 获得第L层（输出层）的相关缓存：
    current_cache = caches[L - 1]  # L-1是因为caches数组下标是从0开始的，如果用L会下标溢出

    # 利用第L层的缓存计算第L层（输出层）的梯度（激活函数sigmoid）并存入梯度字典中
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    # dA^[L-1],dW^[L],db^[L]，先前变量dAL已经求得dA^[L]

    # 从第L-1层开始向前迭代计算梯度（这里迭代长度是0到L-1）
    for l in reversed(range(L - 1)):

        # 缓存从caches列表中倒数第二个（第L-1隐层）开始
        current_cache = caches[l]
        # 求取梯度：dA^[l-1]、dW^[l]、db^[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        # 将所得梯度存入梯度字典（下标l范围0 ~ L-1），需要+1才能指代当前元素
        grads["dA" + str(l)] = dA_prev_temp  # 这里dA是dA^[l-1]，所以不+1
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# # 测试L_model_backward()
# print("==============测试L_model_backward==============")
# AL, Y_assess, caches = testCases.L_model_backward_test_case()
# grads = L_model_backward(AL, Y_assess, caches)
# print("dW1 = " + str(grads["dW1"]))
# print("db1 = " + str(grads["db1"]))
# print("dA1 = " + str(grads["dA1"]))


"""
    五、更新参数
    
        参数更新公式：
            W^[l] = W^[l] - αdW^[l]
            b^[l] = b^[l] - αdb^[l]
"""


def update_parameters(parameters, grads, learning_rate):
    """
    使用梯度下降更新参数
    :param parameters: 未更新前的参数字典
    :param grads: 梯度字典
    :param learning_rate: 学习率
    :return:
        parameters: 更新后的参数值
    """

    # 获取层数（参数字典内含W和b，整除）
    L = len(parameters) // 2

    # 在不同层间迭代更新
    for l in range(L):

        # 更新W^[l]
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        # 更新b^[l]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


# # 测试update_parameters()
# print("==============测试update_parameters==============")
# parameters, grads = testCases.update_parameters_test_case()
# parameters = update_parameters(parameters, grads, 0.1)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


"""
    功能函数已经实现，接下来将这些函数组合，搭建两种不同的神经网络：
        1、两层神经网络
        2、多层神经网络
"""


"""
    1、搭建两层神经网络
        该模型可以概括为：
            INPUT(X) --> LINEAR --> RELU --> LINEAR --> SIGMOID --> OUTPUT(Y^)
"""


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, isPlot=True):
    """
    实现两层神经网络：INPUT --> [LINEAR -> ReLU] --> [LINEAR -> Sigmoid] --> OUTPUT
    :param X: 输入数据，维度(n_x, 样本数)
    :param Y: 标签向量，{0非猫|1猫}，维度(1, 样本数)
    :param layers_dims: 层数向量，维度(n_x, n_h, n_y)
    :param learning_rate: 学习率
    :param num_iterations: 迭代次数
    :param print_cost: 是否打印cost，每100次迭代打印一次
    :param isPlot: 是否绘制iteration/cost图像
    :return:
        parameter: 参数字典，包含W1, b1, W2, b2
    """

    np.random.seed(1)  # 随机数种子
    grads = {}  # 定义空字典供梯度值存储
    costs = []  # 定义空列表供cost值存储
    (n_x, n_h, n_y) = layers_dims  # 获得各层单元个数

    '''
        第一步、初始化参数
    '''

    # 使用函数initialize_parameters()初始化参数
    parameters = initialize_parameters(n_x, n_h, n_y)

    # 从参数字典中获得各个参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    '''
        第二步、迭代
    '''

    for i in range(0, num_iterations):

        # 前向传播

        # 使用linear_activation_forward()函数将线性部分与激活部分合并计算
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        # cost计算，使用compute_cost()
        cost = compute_cost(A2, Y)

        # 反向传播

        # 初始化反向传播，损失函数Loss(A, Y)对输出激活值A^[L]的梯度（笔记公式）：-(y/a + (1-y)/(1-a))
        dA2 = -(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # 反向传播开始
        # 输入：“dA2，cache2，cache1”
        # 输出：“dA1，dW2，db2; 还有dA0（不使用），dW1，db1”。
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        # 将梯度值保存至字典
        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2

        # 更新参数，使用update_parameters()函数
        parameters = update_parameters(parameters, grads, learning_rate)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # 打印cost值，默认print_cost=False不打印
        if i % 100 == 0:

            # 记录成本
            costs.append(cost)

            # 是否打印
            if print_cost:
                print("第", i, "次迭代，cost=", np.squeeze(cost))

    # 迭代结束，根据条件画图，默认isPlot=True画图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('Cost')
        plt.xlabel('Iterations(per hundred)')
        plt.title('Learning Rate = ' + str(learning_rate))
        plt.show()

    # 返回参数字典
    return parameters


"""
    模型预测
        使用two_layer_model()获取到模型的目标参数后，可以使用参数来对测试集进行预测
"""


def predict(X, y, parameters):
    """

    :param X:
    :param y:
    :param parameters:
    :return:
    """

    m = X.shape[1]  # 获取样本数
    n = len(parameters) // 2  # 获取神经网络的层数
    p = np.zeros((1, m))  # 预测输出

    # 使用参数前向传播
    probas, caches = L_model_forward(X, parameters)
    # probas预测概率输出，Y^

    for i in range(0, probas.shape[1]):

        # 预测概率大于0.5
        if probas[0, i] > 0.5:
            # 预测标签为1
            p[0, i] = 1
        # 预测概率小于0.5
        else:
            # 预测标签为0
            p[0, i] = 0

    print("准确度为：", str(float(np.sum((p == y)) / m)))

    # 输出预测标签
    return p


"""
    2、搭建多层神经网络
        模型可以概括为：
            INPUT(X) --> LINEAR --> RELU --> ······ --> LINEAR --> SIGMOID --> OUTPUT(Y^)
"""


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, isPlot=True):
    """
    实现L层神经网络：INPUT(X) --> [LINEAR->RELU] --> ······ --> LINEAR --> SIGMOID --> OUTPUT(Y^)
    :param X: 输入数据，维度(n_x, 样本数)
    :param Y: 标签向量，{0非猫|1猫}，维度(1, 样本数)
    :param layers_dims: 层数向量，维度(n_x, n_h, n_y)
    :param learning_rate: 学习率
    :param num_iterations: 迭代次数
    :param print_cost: 是否打印cost，每100次迭代打印一次
    :param isPlot: 是否绘制iteration/cost图像
    :return:
        parameter: 参数字典，包含W1, b1, W2, b2
    """

    np.random.seed(1)  # 随机数种子
    costs = []  # 创建空列表供存储不同的cost

    # 初始化L层神经网络参数
    parameters = initialize_parameters_deep(layers_dims)

    # 迭代指定次数
    for i in range(0, num_iterations):

        # 前向传播
        # 最终得到A^[L]、缓存
        AL, caches = L_model_forward(X, parameters)

        # cost计算
        cost = compute_cost(AL, Y)

        # 反向传播
        # 计算梯度
        grads = L_model_backward(AL, Y, caches)

        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        # 打印cost值，默认print_cost=False不打印
        if i % 100 == 0:

            # 记录成本
            costs.append(cost)

            # 是否打印
            if print_cost:
                print("第", i, "次迭代，cost=", np.squeeze(cost))

    # 迭代结束，画cost/iterations图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('Cost')
        plt.xlabel('Iterations(per hundred)')
        plt.title('Learning Rate = ' + str(learning_rate))
        plt.show()


    return parameters


"""
    数据集加载
        数据集使用与第二周相同的猫图数据集
        加载和使用
"""


# 加载数据集
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()
# train_set_x_orig：训练集特征输入
# train_set_y：训练集标签
# test_set_x_orig：测试集特征输入
# test_set_y：测试集标签
# classes：样本类别

# 将训练集和测试集的特征输入矩阵维度变平
train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# 将训练集和测试集的特征输入
train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y


"""
    二层神经网络模型


# 训练：
# X输入神经元
n_x = 12288
# 隐层神经元
n_h = 7
# Y^输出神经元
n_y = 1

# 组成参数layers_dims
layers_dims = (n_x, n_h, n_y)

# 训练并获取参数
parameters = two_layer_model(train_x, train_set_y, layers_dims=(n_x, n_h, n_y), num_iterations=2500, print_cost=True, isPlot=True)

# 预测
predictions_train = predict(train_x, train_y, parameters)  # 训练集
predictions_test = predict(test_x, test_y, parameters)  # 测试集
"""


"""
    L层神经网络模型
"""


# 训练
layers_dims = [12288, 20, 7, 5, 1]  # 5-layer model
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True, isPlot=True)

# 预测
pred_train = predict(train_x, train_y, parameters)  # 训练集
pred_test = predict(test_x, test_y, parameters)  # 测试集


"""
    分析
        可以看看哪些图片在L层模型中被错误预测，导致准确率没有提高
"""


def print_mislabeled_images(classes, X, y, p):
    """
    绘制预测错误的图像
    :param classes: 类别
    :param X: 输入特征
    :param y: 实际标签
    :param p: 预测
    """

    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i+1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction:" + classes[int(p[0, index])].decode("utf-8") + "\n Class:" + classes[y[0, index]].decode("utf-8"))

    plt.show()


print_mislabeled_images(classes, test_x, test_y, pred_test)


"""
    可得出误判原因：
    
        · 猫身体在不同的位置
        · 猫出现在相似颜色的背景下
        · 不同的猫的颜色和品种
        · 相机角度
        · 图片亮度
        · 比例变化
"""


"""
    选做任务
    
        使用一张特定角度图片进行识别
        位置在此工程根目录
"""


# # # START CODE HERE # #
# my_image = "my_image.jpg"  # change this to the name of your image file
# my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)
# # # END CODE HERE # #
#
# fname = "images/" + my_image
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((num_px * num_px * 3, 1))
# my_predicted_image = predict(my_image, my_label_y, parameters)
#
# plt.imshow(image)
# print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

