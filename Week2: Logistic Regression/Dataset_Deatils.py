import numpy as np
import h5py
import matplotlib.pyplot as plt
from lr_utils import load_dataset

# 1、HDF5的读取
f = h5py.File('datasets/train_catvnoncat.h5', 'r')
# 加载数据集训练集图片数据、训练集标签、测试集图片数据、测试集标签、标签含义（0：non-cat；1：cat）
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

index = 25
# 2、查看第index+1个图片
# plt.imshow(train_set_x_orig[index])
# print("train_set_y=" + str(train_set_y))  # 输出训练集的标签
# plt.show()


# 3、打印第index+1个训练样例的标签值
# 使用np.squeeze(train_set_y[:,index])是为了压缩维度
# 压缩前为[1]，压缩后为1
# 因为使用的是切片，得到的还是一个数组而不是整型，所以使用squeeze将其转换为整型才可以解码输出。
# print("squeeze前：" + str(train_set_y[:, index]) + "（np数组）\nsqueeze后：" + str(np.squeeze(train_set_y[:, index])), '（整型数据）')
# 压缩后才能进行解码操作
# print("y=", str(train_set_y[:, index]), ", it is a", classes[np.squeeze(train_set_y[:, index])].decode("utf-8"), "picture")


m_train = train_set_y.shape[1]  # 训练集图片数量
m_test = test_set_y.shape[1]  # 测试集图片数量
num_px = train_set_x_orig.shape[1]  # 训练、测试集图片的宽度和高度（均为64x64）

# 现在看一看我们加载的东西的具体情况
# print("训练集的数量: m_train = " + str(m_train))  # m_train = 209
# print("测试集的数量: m_test = " + str(m_test))  # m_test = 50
# print("每张图片的宽/高: num_px = " + str(num_px))  # num_px = 64
# print("每张图片的大小: (" + str(num_px) + ", " + str(num_px) + ", 3)")  # (64, 64, 3)
# print("训练集_图片的维数: " + str(train_set_x_orig.shape))  # (209, 64, 64, 3)
# print("训练集_标签的维数: " + str(train_set_y.shape))  # (1, 209)
# print("测试集_图片的维数: " + str(test_set_x_orig.shape))  # (50, 64, 64, 3)
# print("测试集_标签的维数: " + str(test_set_y.shape))  # (1, 50)


# 4、降维
# X_flatten = X.reshape(X.shape[0]，-1).T  # 将(209, 64, 64, 3)拉伸为(209, 64*64*3)后转置为(64*64*3, 209)
# -1表示模糊控制，比如人reshape（2, -1）固定2行 多少列不确定
# X.shape为(209, 64, 64, 3)
# 使X变成209行，(64*64*3)列，每行 (209*64*64*3)/209 = 64*64*3 个元素
# 先确定除了参数-1之外的其他参数，然后通过(总参数的计算)/(确定除了参数-1之外的其他参数)=该位置应该是多少的参数

# 将训练集的维度(209, 64, 64, 3)降低并转置。
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# 将测试集的维度(50, 64, 64, 3)降低并转置。
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# 经过这两步操作把数组变为(12288, 209)的矩阵，每1列代表一张图片

# 降维后
print("训练集降维最后的维度： " + str(train_set_x_flatten.shape))
print("训练集_标签的维数 : " + str(train_set_y.shape))
print("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
print("测试集_标签的维数 : " + str(test_set_y.shape))


# 5、数据集归一化
# 为了表示彩色图像，必须为每个像素指定红色，绿色和蓝色通道（RGB），因此像素值实际上是从0到255范围内的三个数字的向量。
# 机器学习中一个常见的预处理步骤是对数据集进行居中和标准化，这意味着可以减去每个示例中整个numpy数组的平均值，然后将每个示例除以整个numpy数组的标准偏差。
# 对于图片数据集，它更简单方便，几乎可以将数据集的每一行除以255（像素通道的最大值），因为在RGB中不存在比255大的数据，所以我们可以放心的除以255，让标准化的数据位于[0,1]之间
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255
# 该步针对训练集和测试集进行操作
