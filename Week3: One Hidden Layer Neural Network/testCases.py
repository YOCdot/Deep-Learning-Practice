# 测试用例

import numpy as np


def layer_sizes_test_case():
    """
    层大小测试，用于测试函数layer_sizes()
    :return X_assess: 一个(5, 3)维的高斯分布随机数组
    :return Y_assess: 一个(2, 3)维的高斯分布随机数组
    """
    np.random.seed(1)
    X_assess = np.random.randn(5, 3)
    Y_assess = np.random.randn(2, 3)
    return X_assess, Y_assess


def initialize_parameters_test_case():
    """
    初始化参数测试，用于测试函数initialize_parameters()
    :return n_x: 返回输入层神经元个数
    :return n_h: 返回隐层神经元个数
    :return n_y: 返回输出层神经元个数
    """
    n_x, n_h, n_y = 2, 4, 1
    return n_x, n_h, n_y


def forward_propagation_test_case():
    """
    前向传播测试用例。用于测试函数forward_propagation()
    :return X_assess: 输入数据
    :return parameters: 参数字典
    """
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)

    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
                                  [-0.02136196, 0.01640271],
                                  [-0.01793436, -0.00841747],
                                  [0.00502881, -0.01245288]]),
                  'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
                  'b1': np.array([[0.],
                                  [0.],
                                  [0.],
                                  [0.]]),
                  'b2': np.array([[0.]])}

    return X_assess, parameters


def compute_cost_test_case():
    """
    cost计算测试用例。用于测试函数compute_cost()
    :return a2: sigmoid激活后的值
    :return Y_assess: "True"标签向量，维度为（1, 样本个数）
    :return parameters: 参数矩阵
    """
    np.random.seed(1)
    Y_assess = np.random.randn(1, 3)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
                                  [-0.02136196, 0.01640271],
                                  [-0.01793436, -0.00841747],
                                  [0.00502881, -0.01245288]]),
                  'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
                  'b1': np.array([[0.],
                                  [0.],
                                  [0.],
                                  [0.]]),
                  'b2': np.array([[0.]])}

    a2 = (np.array([[0.5002307, 0.49985831, 0.50023963]]))

    return a2, Y_assess, parameters


def backward_propagation_test_case():
    """
    反向传播测试用例，测试函数backward_propagation()。
    :return parameters: 参数字典
    :return cache: 梯度字典
    :return X_assess: 输入样本矩阵，维度为(2, 样本个数)
    :return Y_assess: "True"标签向量，维度为（1, 样本个数）
    """
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = np.random.randn(1, 3)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
                                  [-0.02136196, 0.01640271],
                                  [-0.01793436, -0.00841747],
                                  [0.00502881, -0.01245288]]),
                  'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
                  'b1': np.array([[0.],
                                  [0.],
                                  [0.],
                                  [0.]]),
                  'b2': np.array([[0.]])}

    cache = {'A1': np.array([[-0.00616578, 0.0020626, 0.00349619],
                             [-0.05225116, 0.02725659, -0.02646251],
                             [-0.02009721, 0.0036869, 0.02883756],
                             [0.02152675, -0.01385234, 0.02599885]]),
             'A2': np.array([[0.5002307, 0.49985831, 0.50023963]]),
             'Z1': np.array([[-0.00616586, 0.0020626, 0.0034962],
                             [-0.05229879, 0.02726335, -0.02646869],
                             [-0.02009991, 0.00368692, 0.02884556],
                             [0.02153007, -0.01385322, 0.02600471]]),
             'Z2': np.array([[0.00092281, -0.00056678, 0.00095853]])}
    return parameters, cache, X_assess, Y_assess


def update_parameters_test_case():
    """
    参数更新测试用例。测试函数update_parameters()
    :return parameters: 参数矩阵
    :return grads: 梯度矩阵
    """
    parameters = {'W1': np.array([[-0.00615039, 0.0169021],
                                  [-0.02311792, 0.03137121],
                                  [-0.0169217, -0.01752545],
                                  [0.00935436, -0.05018221]]),
                  'W2': np.array([[-0.0104319, -0.04019007, 0.01607211, 0.04440255]]),
                  'b1': np.array([[-8.97523455e-07],
                                  [8.15562092e-06],
                                  [6.04810633e-07],
                                  [-2.54560700e-06]]),
                  'b2': np.array([[9.14954378e-05]])}

    grads = {'dW1': np.array([[0.00023322, -0.00205423],
                              [0.00082222, -0.00700776],
                              [-0.00031831, 0.0028636],
                              [-0.00092857, 0.00809933]]),
             'dW2': np.array([[-1.75740039e-05, 3.70231337e-03, -1.25683095e-03,
                               -2.55715317e-03]]),
             'db1': np.array([[1.05570087e-07],
                              [-3.81814487e-06],
                              [-1.90155145e-07],
                              [5.46467802e-07]]),
             'db2': np.array([[-1.08923140e-05]])}
    return parameters, grads


def nn_model_test_case():
    """
    函数整合测试用例。测试函数nn_model()
    :return X_assess: 输入
    :return Y_assess: 输出
    """
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = np.random.randn(1, 3)
    return X_assess, Y_assess


def predict_test_case():
    """
    预测测试用例。
    :return parameters: 参数字典
    :return X_assess: 输入
    """
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    parameters = {'W1': np.array([[-0.00615039, 0.0169021],
                                  [-0.02311792, 0.03137121],
                                  [-0.0169217, -0.01752545],
                                  [0.00935436, -0.05018221]]),
                  'W2': np.array([[-0.0104319, -0.04019007, 0.01607211, 0.04440255]]),
                  'b1': np.array([[-8.97523455e-07],
                                  [8.15562092e-06],
                                  [6.04810633e-07],
                                  [-2.54560700e-06]]),
                  'b2': np.array([[9.14954378e-05]])}
    return parameters, X_assess
