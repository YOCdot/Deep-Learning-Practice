"""
文件说明：

    testCases.py
    测试用例

        包含：
            1、正向传播：线性部分Z——linear_forward()函数测试用例
            2、正向传播：线性+激活部分g(Z)——linear_activation_forward()函数测试用例
            3、正向传播：L层神经网络——L_model_forward()测试用例
            4、代价计算：总体代价计算Loss(A, Y)——compute_cost()测试用例
            5、反向传播：线性部分Z——linear_backward()测试用例
            6、反向传播：线性+激活部分g(Z)——linear_activation_backward()函数测试用例
            7、反向传播：L层神经网络——L_model_backward()测试用例
            8、参数更新：L层神经网络层间迭代更新——update_parameters()测试用例
"""


import numpy as np


def linear_forward_test_case():
    """
    【LINEAR】线性部分
    linear_forward()函数测试样例。

    :return:
        X = np.array([[ 1.62434536, -0.61175641],
           [-0.52817175, -1.07296862],
           [ 0.86540763, -2.3015387 ]])
        W = np.array([[ 1.74481176, -0.7612069 ,  0.3190391 ]])
        b = np.array([[-0.24937038]]))
    """

    np.random.seed(1)

    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)

    return A, W, b


def linear_activation_forward_test_case():
    """
    【LINEAR -> ACTIVATION】线性激活部分
    linear_activation_forward()函数测试样例。

    :return:
        X = np.array([[-0.41675785, -0.05626683],
           [-2.1361961 ,  1.64027081],
           [-1.79343559, -0.84174737]]),
        W = np.array([[ 0.50288142, -1.24528809, -1.05795222]]),
        b = np.array([[-0.90900761]])
    """

    np.random.seed(2)
    A_prev = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    return A_prev, W, b


def L_model_forward_test_case():
    """
    L层神经网络前向传播
    L_model_forward()测试用例。

    :return:
        X = np.array([[ 1.62434536, -0.61175641],
                    [-0.52817175, -1.07296862],
                    [ 0.86540763, -2.3015387 ],
                    [ 1.74481176, -0.7612069 ]]),
        parameters = {
            'W1': array([[ 0.3190391 , -0.24937038,  1.46210794, -2.06014071],
                        [-0.3224172 , -0.38405435,  1.13376944, -1.09989127],
                        [-0.17242821, -0.87785842,  0.04221375,  0.58281521]]),
            'b1': array([[-1.10061918],
                        [ 1.14472371],
                        [ 0.90159072]]),
            'W2': array([[ 0.50249434,  0.90085595, -0.68372786]]),
            'b2': array([[-0.12289023]])}
    """

    np.random.seed(1)
    X = np.random.randn(4, 2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return X, parameters


def compute_cost_test_case():
    """
    代价计算
    compute_cost()测试用例。
    """

    Y = np.asarray([[1, 1, 1]])
    aL = np.array([[.8, .9, 0.4]])

    return Y, aL


def linear_backward_test_case():
    """
    线性部分Z反向传播
    linear_backward()测试用例。
    :return:
        z, linear_cache = (np.array([[ 1.62434536, -0.61175641]]),
                        (np.array([[-0.52817175, -1.07296862],
                                [ 0.86540763, -2.3015387 ],
                                [ 1.74481176, -0.7612069 ]]),
                        np.array([[ 0.3190391 , -0.24937038,  1.46210794]]),
                        array([[-2.06014071]])))
    """

    np.random.seed(1)

    dZ = np.random.randn(1, 2)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    linear_cache = (A, W, b)
    return dZ, linear_cache



def linear_activation_backward_test_case():
    """
    【LINEAR -> ACTIVATION】线性+激活部分反向传播
    linear_activation_backward()测试样例。
    :return:
        dA, linear_activation_cache = (np.array([[-0.41675785, -0.05626683]]),
                                    ((array([[-2.1361961 ,  1.64027081],
                                            [-1.79343559, -0.84174737],
                                            [ 0.50288142, -1.24528809]]),
                                    np.array([[-1.05795222, -0.90900761,  0.55145404]]),
                                    np.array([[2.29220801]])),
                                    array([[ 0.04153939, -1.11792545]])))
    """
    np.random.seed(2)
    dA = np.random.randn(1, 2)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    Z = np.random.randn(1, 2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)

    return dA, linear_activation_cache


def L_model_backward_test_case():
    """
    反向计算整个模型
    L_model_backward()测试用例。

    :return:
        AL = np.array([[1.78862847, 0.43650985]]),
        Y = np.array([[1, 0]]),
        caches =
            (
            ((
            A1 = array([[ 0.09649747, -1.8634927 ],
                        [-0.2773882 , -0.35475898],
                        [-0.08274148, -0.62700068],
                        [-0.04381817, -0.47721803]]),
            W1 = array([[-1.31386475,  0.88462238,  0.88131804,  1.70957306],
                        [ 0.05003364, -0.40467741, -0.54535995, -1.54647732],
                        [ 0.98236743, -1.10106763, -1.18504653, -0.2056499 ]]),
            b1 = array([[ 1.48614836],
                        [ 0.23671627],
                        [-1.02378514]])),
            Z1 = array([[-0.7129932 ,  0.62524497],
                        [-0.16051336, -0.76883635],
                        [-0.23003072,  0.74505627]]
            )),

            ((
            A2 = array([[ 1.97611078, -1.24412333],
                        [-0.62641691, -0.80376609],
                        [-2.41908317, -0.92379202]]),
            W2 = array([[-1.02387576,  1.12397796, -0.13191423]]),
            b2 = array([[-1.62328545]])),
            Z2 = array([[ 0.64667545, -0.35627076]])
            ))
            )
   """

    np.random.seed(3)

    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4, 2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    Z1 = np.random.randn(3, 2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3, 2)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    Z2 = np.random.randn(1, 2)
    linear_cache_activation_2 = ((A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)

    return AL, Y, caches


def update_parameters_test_case():
    """
    参数更新，L层神经网络层间迭代更新
    update_parameters()测试用例

    :return:
        ({'W1': array([[-0.41675785, -0.05626683, -2.1361961 ,  1.64027081],
                    [-1.79343559, -0.84174737,  0.50288142, -1.24528809],
                    [-1.05795222, -0.90900761,  0.55145404,  2.29220801]]),
        'b1': array([[ 0.04153939],
                    [-1.11792545],
                    [ 0.53905832]]),
        'W2': array([[-0.5961597 , -0.0191305 ,  1.17500122]]),
        'b2': array([[-0.74787095]])},
        {'dW1': array([[ 1.78862847,  0.43650985,  0.09649747, -1.8634927 ],
                    [-0.2773882 , -0.35475898, -0.08274148, -0.62700068],
                    [-0.04381817, -0.47721803, -1.31386475,  0.88462238]]),
        'db1': array([[0.88131804],
                    [1.70957306],
                    [0.05003364]]),
        'dW2': array([[-0.40467741, -0.54535995, -1.54647732]]),
        'db2': array([[0.98236743]])})
    """

    np.random.seed(2)

    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3, 4)
    db1 = np.random.randn(3, 1)
    dW2 = np.random.randn(1, 3)
    db2 = np.random.randn(1, 1)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return parameters, grads
