import numpy as np


# # 与门实现
# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = x1*w1 + x2*w2
#     if tmp <= theta:
#         return 0
#     elif tmp > theta:
#         return 1


# 使用权重和偏置的实现
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x*w) + b
    if tmp < 0:
        return 0
    else:
        return 1


# 使用权重和偏置的实现    与非门  修改对应 w 和 b
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x*w) + b
    if tmp < 0:
        return 0
    else:
        return 1


# 使用权重和偏置的实现    或门  修改对应 w 和 b
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0.2
    tmp = np.sum(x*w) + b
    if tmp < 0:
        return 0
    else:
        return 1


# 异或门的实现
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


