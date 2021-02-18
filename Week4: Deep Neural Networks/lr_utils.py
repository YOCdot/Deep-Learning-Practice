"""
文件说明：

    lr_utils.py
    数据集加载库

        包含：
            数据集加载函数——load_dataset()

"""

import numpy as np
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # 训练集特征
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 训练集标签

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # 测试集特征
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # 测试集标签

    classes = np.array(test_dataset["list_classes"][:])  # 类别列表: [b'non-cat' b'cat']
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
