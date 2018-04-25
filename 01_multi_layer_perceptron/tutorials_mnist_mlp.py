#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
@file: tutorials_artificial_neural_network.py
"""
import os
import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# 读取MNIST数据集
def load_data():
    # 获取当前.py文件的绝对路径和文件名
    dir_path, file_name = os.path.split(os.path.abspath(sys.argv[0]))

    # 获取当前文件夹的上级目录
    par_path = os.path.dirname(dir_path)

    # 获取MNIST数据集的地址
    datapath = par_path + "\\00_data_set" + "\\MNIST_data"

    # 加载MNIST数据集,并提取train/test数据
    print('1.Loading data set...')
    dataset = input_data.read_data_sets(datapath, one_hot=True)

    train_datas = dataset.train.images
    train_label = dataset.train.labels
    test_datas = dataset.test.images
    test_label = dataset.test.labels

    return train_datas, train_label, test_datas, test_label


# 分类算法--人工神经网络
class Ann(object):
    # 人工神经网络初始化
    def __init__(self, sizes):
        """
        :param sizes: list类型,储存每层神经网络的神经元数目; 假如sizes=[2,3,2]
                      表示输入层有2个神经元/隐藏层有3个神经元/输出层有2个神经元
        """
        self.num_layers = len(sizes)
        self.sizes = sizes

        # 生成隐藏层和输出层的biases,维度为(n[l],1)(n(l)为第l层的神经元个数),取数范围:正态分布的随机样本数
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # 生成隐藏层和输出层的weight,维度为(n(l),n(l-1)),取数范围:正态分布的随机样本数
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # 梯度下降算法迭代训练人工神经网络
    def GradientDescentOpt(self, datas, label, epochs, mini_batch_size, rate):
        """
        :param datas: 输入的训练数据
        :param label: 输入的数据标签
        :param epochs: 迭代次数
        :param mini_batch_size: mini_batch样本个数
        :param rate: 学习率
        """
        num = len(datas)
        for j in range(epochs):
            # 随机打散训练集的排序
            shuffle_idx = np.random.permutation(num)
            shuffle_x = datas[shuffle_idx, :]
            shuffle_y = label[shuffle_idx, :]

            # 将训练集划分若干mini_batches,每个mini_batch进行神经网络训练
            for k in range(0, num, mini_batch_size):
                batch_xs = shuffle_x[k:k+mini_batch_size]
                batch_ys = shuffle_y[k:k+mini_batch_size]
                self.update_mini_batch(batch_xs, batch_ys, rate)

            # 每轮训练后,神经网络的预测准确率
            print("***** Epoch {0}:  {1}/{2}".format(j, self.evaluate(datas, label), num))

    # mini_batch更新参数w和b
    def update_mini_batch(self, batch_xs, batch_ys, rate):
        """
        :param batch_xs: mini_batch训练数据
        :param batch_ys: mini_batch数据标签
        :param rate: 学习率
        """

        # 前向计算输出
        z, a = self.feed_forward(batch_xs)

        # 后向计算梯度
        dw, db = self.back_prop(batch_xs, batch_ys, z, a)

        # 更新参数w和b
        self.weights = [w-rate*nw for w, nw in zip(self.weights, dw)]
        self.biases = [b-rate*nb for b, nb in zip(self.biases, db)]

    # 前向计算输出
    def feed_forward(self, batch_xs):
        z = [batch_xs.transpose()]
        a = [batch_xs.transpose()]
        for i in range(self.num_layers-1):
            w = self.weights[i]
            b = self.biases[i]
            z_i = np.dot(w, a[i]) + b
            a_i = sigmoid(z_i)
            z.append(z_i)
            a.append(a_i)

        return z, a

    # 后向计算梯度
    def back_prop(self, x, y, z, a):

        dz = [np.zeros(s.shape) for s in z]
        dw = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        idx = list(range(self.num_layers-1))
        for i in reversed(idx):
            if i == idx[-1]:
                dz[i+1] = a[i+1] - y.transpose()
            else:
                tmp0 = np.dot(self.weights[i+1].transpose(), dz[i+2])
                tmp1 = sigmoid_prime(z[i+1])
                dz[i+1] = np.multiply(tmp0, tmp1)

            dw[i] = np.dot(dz[i+1], a[i].transpose())/x.shape[0]
            db[i] = np.sum(dz[i+1], axis=1, keepdims=True)/x.shape[0]

        return dw, db

    # 测试样本结果
    def evaluate(self, test_xs, test_ys):
        tmp_z, tmp_a = self.feed_forward(test_xs)
        pred_a = tmp_a[-1].transpose()
        num = 0
        for x, y in zip(pred_a, test_ys):
            x_idx = np.where(x == np.max(x))
            y_idx = np.where(y == np.max(y))
            if x_idx == y_idx:
                num += 1

        return num


# 激活函数--sigmoid
def sigmoid(z):
    z_res = 1.0/(1.0+np.exp(-z))

    return z_res


# 激活函数的导数
def sigmoid_prime(z):
    z_derivative = sigmoid(z) * (1-sigmoid(z))

    return z_derivative


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()

    # 初始化人工神经网络
    ann_net = Ann([784, 30, 10])

    # 人工神经网络训练和预测
    print('开始训练神经网络......')
    ann_net.GradientDescentOpt(x_train, y_train, 10, 100, 3.0)
    
    # 测试样本结果
    acc_num = ann_net.evaluate(x_test, y_test)
    accuracy = acc_num/y_test.shape[0]
    print('测试样本分类准确率:', accuracy)
