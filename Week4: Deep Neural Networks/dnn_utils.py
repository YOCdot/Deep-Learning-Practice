"""
文件说明：

    dnn_utils.py
    dnn函数库

        包含：
            1、Sigmoid激活函数
            2、Sigmoid激活函数反向传播（求导函数，获得梯度）
            3、ReLU激活函数
            4、ReLU激活函数反向传播（求导函数，获得梯度）
"""


import numpy as np


def sigmoid(Z):
    """
    实现sigmoid激活函数（NumPy）。
    参数:
    Z -- 任意形状的numpy数组
    返回值:
    A -- sigmoid(z)的值，与Z维度一致。
    cache -- 返回Z，用于反向传播过程中。
    """

    A = 1 / (1 + np.exp(-Z))

    cache = Z

    return A, cache


def sigmoid_backward(dA, cache):
    """
    实现一个sigmoid单元的反向传播。

    参数:
    dA -- 激活后的梯度，任意维度
    cache -- 'Z' where we store for computing backward propagation efficiently

    返回值:
    dZ -- 代价函数相对于Z的梯度
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def relu(Z):
    """
    实现ReLU激活函数。

    参数:
    Z -- Output of the linear layer, of any shape

    返回值:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    实现一个ReLu单元的反向传播。

    参数:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    返回值:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ
