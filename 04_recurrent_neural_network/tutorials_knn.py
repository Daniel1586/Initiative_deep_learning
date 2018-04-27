#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
@file: tutorials_knn.py
"""
import csv
import random
import operator
import numpy as np
import pandas as pd


# 读取样本数据,最后1列为类别
def load_data(file):
    dataset = pd.read_csv(file)

    return dataset


# 数据集划分训练集和测试集
# 返回数据类型为n维数组格式
def split_dataset(dataset, test_size=1):
    cols = dataset.columns.tolist()
    data_idx = dataset.index.tolist()
    test_idx = random.sample(data_idx, test_size)
    train_idx = [i for i in data_idx if i not in test_idx]

    # 计算训练集, 转化为n维数组格式
    trainset = dataset.iloc[train_idx]
    train_datas = trainset.values[:, 0:len(cols)-1]
    train_label = trainset.values[:, -1:]

    # 计算测试集, 转化为n维数组格式
    testset = dataset.iloc[test_idx]
    test_datas = testset.values[:, 0:len(cols)-1]
    test_label = testset.values[:, -1:]

    return train_datas, train_label, test_datas, test_label


# 分类算法--KNN
class KNearestNeighbor(object):

    def __init__(self):
        pass

    # 计算距离度量
    @staticmethod
    def calc_distance(traindata, testdata):

        distance = np.sqrt(np.sum(np.square(traindata-testdata)))

        return distance

    # 计算距离最近的k个样本
    def get_neighbors(self, datas, label, test, k):
        distances = []
        # 测试样本和训练集中每个样本计算距离
        for i in range(len(datas)):
            dist = self.calc_distance(datas[i], test)
            distances.append((datas[i], dist, label[i]))
        distances.sort(key=operator.itemgetter(1))

        neighbors = []
        for j in range(k):
            neighbors.append(distances[j][2])

        return neighbors

    # 按多数投票结果归类
    @staticmethod
    def get_class(neighbors):
        classes = {}
        for i in range(len(neighbors)):
            class_i = neighbors[i][0]
            if class_i in classes.keys():
                classes[class_i] += 1
            else:
                classes[class_i] = 1

        # 类别数目降序排列
        class_sorted = sorted(classes.items(), key=operator.itemgetter(1), reverse=True)
        class_vote = class_sorted[0][0]

        return class_vote

    # 计算测试样本准确率
    @staticmethod
    def get_accuracy(test_label, predictions):
        correct = 0
        for i in range(len(test_label)):
            if test_label[i][-1] == predictions[i]:
                correct += 1
        print('预测准确率: {0}/{1}'.format(correct, len(test_label)))
        acc = correct/len(test_label)

        return acc

    # 测试样本分类
    def run(self, train_datas, train_label, test_datas, test_label, k=3):
        predictions = []
        # 遍历测试样本
        for i in range(len(test_datas)):
            neighbors = self.get_neighbors(train_datas, train_label, test_datas[i], k)
            test_class = self.get_class(neighbors)
            predictions.append(test_class)

        accuracy = self.get_accuracy(test_label, predictions)
        print('Accuracy: ' + repr(accuracy*100) + '%')


if __name__ == "__main__":
    filename = "dataset_iris.csv"
    data_set = load_data(filename)
    x_train, y_train, x_test, y_test = split_dataset(data_set, test_size=45)

    clf = KNearestNeighbor()
    clf.run(x_train, y_train, x_test, y_test, k=3)
