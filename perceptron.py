# coding:utf-8
import numpy as np
import pandas as pd
"""
简单感知机学习算法

感知机模型:f(x)=sign(w*x+b)
算法策略:损失函数:L(w,b)=y(w*x+b)
学习算法:梯度下降法
w = w + eta*y*x
b = b + eta*y

数据来源：
《统计学习方法》 李航 例2.1 P29
@Python
"""


class Perceptron:
    """Perceptron

    Methods
    ---------

    __init__: 构造函数
    isError: 判断是否为误分类点
    updateWeights: 更新参数
    train: 训练数据


    """
    def __init__(self, eta, w0, b0, iterMax, data):
    	"""
        eta: 学习率，w0：权重向量w的初始值，b0:偏置b的初始值，iterMax:最大循环次数
        """
        self.eta = eta
        self.w = w0
        self.b = b0
        self.iterMax = iterMax
        self.data = data

    def isError(self, record):
        if (np.dot(self.w, record[0:-1]) + self.b)*record[-1] > 0:
            return False
        else:
            return True

    def updateWeights(self, record_err):
        self.w = self.w + self.eta * record_err[0:-1] * record_err[-1]
        self.b = self.b + self.eta * record_err[-1]
        return

    def train(self):
        n = len(self.data)
        flag = True  # Ture 仍包含误分类点；False 没有误分类点
        iterNum = 0
        """
        停止条件：
        训练集中没有误分类点 或 循环次数超限
        """
        while flag and iterNum < self.iterMax:
            iterNum += 1
            for i in range(n):
                if self.isError(data.values[i]):
                    self.updateWeights(data.values[i])
                    flag = True
                else:
                    flag = False
        return (self.w, self.b)

if __name__ == "__main__":
    data = pd.DataFrame()
    data["x1"] = np.array([3, 4, 1])
    data["x2"] = np.array([3, 3, 1])
    data["y"] = np.array([1, 1, -1])
    n = len(data)
    p = Perceptron(1, np.zeros(2), 0, 10, data)
    result = p.train()
    print str(result)
