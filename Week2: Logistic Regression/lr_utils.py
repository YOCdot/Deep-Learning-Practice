import numpy as np
import h5py
    
    
def load_dataset():

    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")  # 加载训练集
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # 训练集内图像数据
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 训练集内图像对应的标签。0不是猫，1是猫。

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")  # 加载测试集
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # 测试集内图像数据
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # 测试集内图像对应的标签。0不是猫，1是猫。

    # 保存的是以bytes类型保存的两个字符串数据，数据为：[b’non-cat’ b’cat’]。
    classes = np.array(test_dataset["list_classes"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  # 转换成行
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))  # 转换为行
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
