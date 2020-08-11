import numpy as np
import matplotlib.pyplot as plt


# 跃阶函数的实现
def step_function(x):
    # y = x > 0
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


# 针对溢出 做出对应改进
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y1 = exp_a / sum_exp_a
    return y1


x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # 指定y轴的范围
plt.show()






