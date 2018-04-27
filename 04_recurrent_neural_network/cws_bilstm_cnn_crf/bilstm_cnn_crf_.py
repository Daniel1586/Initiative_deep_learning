#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
@file: bilstm_cnn_crf.py
"""
import os,re
import codecs
import pickle
import time

import gc
import numpy as np
np.random.seed(1111)

import gensim
import random

import keras
# keras.backend.clear_session()

from keras.layers import *
from keras.models import *
from keras_contrib.layers import CRF

from keras import backend as K

from keras.utils import plot_model
from keras.utils import np_utils

from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint

from keras.models import model_from_json


class Documents(object):
    def __init__(self, chars, label, index):
        self.chars = chars
        self.label = label
        self.index = index


# 训练模型 保存weights
def process_train(corpus_path, nb_epoch, base_model_weight=None):

    # 1.读取corpus语料下,语料文件夹下各语料文件地址
    raw_train_file = [corpus_path + os.sep + folder + os.sep + file
                      for folder in os.listdir(corpus_path)
                      for file in os.listdir(corpus_path + os.sep + folder)]

    # 2.读取每个语料文件生成train.data, 利用train.data生成训练数据原始格式
    process_data(raw_train_file, 'train.data')
    train_docs = create_docs('train.data')
    print("***** 按标点切分后训练数据长度: ", len(train_docs))

    # 3.根据语料数据生成词典
    lexicon, lexicon_reverse = get_lexicon(train_docs)
    print("***** 生成词典长度: ", len(lexicon))

    # 4.load预训练词向量
    embedding_model = gensim.models.Word2Vec.load(r'model_conll_law.m')
    embedding_size = embedding_model.vector_size
    print("***** 每个词的词向量维度: ", embedding_size)
    embedding_weights = load_embedding(embedding_model, embedding_size, lexicon_reverse)
    print("***** 训练好的词向量维度: ", embedding_weights.shape)

    # 0 为padding的label
    label_2_index = {'Pad': 0, 'B': 1, 'M': 2, 'E': 3, 'S': 4, 'Unk': 5}
    index_2_label = {0: 'Pad', 1: 'B', 2: 'M', 3: 'E', 4: 'S', 5: 'Unk'}
    # 将训练数据的原始格式转换为字典中的下标表示
    train_data_list, train_label_list, train_index_list = create_matrix(train_docs, lexicon, label_2_index)

    print(len(train_data_list), len(train_label_list), len(train_label_list))
    print(train_data_list[0])
    print(train_label_list[0])

    max_len = max(map(len, train_data_list))
    print('maxlen:', max_len)

    train_data_array, train_label_list_padding = padding_sentences(train_data_list, train_label_list, max_len)
    print(train_data_array.shape)
    print(train_data_array[0])

    train_label_array = np_utils.to_categorical(train_label_list_padding, len(label_2_index)). \
        reshape((len(train_label_list_padding), len(train_label_list_padding[0]), -1))
    print(train_label_array.shape)
    print(train_label_array[0])

    # model
    model = Bilstm_CNN_Crf(max_len, len(lexicon), len(label_2_index), embedding_weights)
    print(model.input_shape)
    print(model.output_shape)

    plot_model(model, to_file='bilstm_cnn_crf_model.png', show_shapes=True, show_layer_names=True)

    if base_model_weight != None and os.path.exists(base_model_weight) == True:
        model.load_weights(base_model_weight)

    hist = model.fit(train_data_array, train_label_array, batch_size=256, epochs=nb_epoch, verbose=1)

    # model.load_weights('best_val_model.hdf5')

    '''
    test_y_pred=model.predict(train_data_array,batch_size=512,verbose=1)
    pred_label=np.argmax(test_y_pred,axis=2)
    print(pred_label[0])

    '''
    score = model.evaluate(train_data_array, train_label_array, batch_size=512)
    print(score)

    # save model
    model.save_weights('train_model.hdf5')

    # save lexicon
    pickle.dump([lexicon, lexicon_reverse, max_len, index_2_label], open('lexicon.pkl', 'wb'))


# 读取每个语料文件生成train.data,每个汉字标记类型{Single,Begin,Middle,End}
def process_data(file_list, new_file):
    res = codecs.open(new_file, 'w', 'utf-8')
    for file in file_list:
        with codecs.open(file, 'r', 'utf-8') as fs:
            lines = fs.readlines()
            for line in lines:
                word_list = line.strip().split()
                for word in word_list:
                    if len(word) == 1:
                        res.write(word + '\tS\n')
                    else:
                        res.write(word[0] + '\tB\n')
                        for w in word[1:-1]:
                            res.write(w + '\tM\n')
                        res.write(word[-1] + '\tE\n')
                res.write('\n')
    res.close()


# 将训练语料中的句子按标点切分,避免某些行过长,难以整个序列优化
def create_docs(file_name):
    docs = []
    chars, label = [], []

    with codecs.open(file_name, 'r', 'utf-8') as f:
        index = 0
        for line in f:
            line = line.strip()
            if len(line) == 0:
                if len(chars) != 0:
                    docs.append(Documents(chars, label, index))
                    chars = []
                    label = []
                index += 1
            else:
                pieces = line.strip().split()
                chars.append(pieces[0])
                label.append(pieces[1])

                if pieces[0] in ['。', '，', '；']:
                    docs.append(Documents(chars, label, index))
                    chars = []
                    label = []

        if len(chars) != 0:
            docs.append(Documents(chars, label, index))

    return docs


# 生成词典,词典标号根据汉字(标点/数字/字母)出现次数标记,次数越多,标记越小
def get_lexicon(all_docs):
    chars = {}
    for doc in all_docs:
        for char in doc.chars:
            chars[char] = chars.get(char, 0) + 1

    # 按汉字(标点/数字/字母)出现次数降序排列
    sorted_chars = sorted(chars.items(), key=lambda x: x[1], reverse=True)

    # 词典下标从1开始,0位保留,用作补长位
    lexicon = dict([(item[0], index+1) for index, item in enumerate(sorted_chars)])
    lexicon_reverse = dict([(index+1, item[0]) for index, item in enumerate(sorted_chars)])

    return lexicon, lexicon_reverse


# load预训练词向量,前后增加1个汉字的权重
def load_embedding(model, size, lexicon_reverse):
    weights = np.zeros((len(lexicon_reverse)+2, size))
    for i in range(len(lexicon_reverse)):
        weights[i+1] = model[lexicon_reverse[i+1]]
    weights[-1] = np.random.uniform(-1, 1, size)

    return weights


# 将训练数据的原始格式转换为字典中的下标表示
def create_matrix(docs, lexicon, lab2idx):
    datas_list = []
    label_list = []
    index_list = []
    for doc in docs:
        datas_tmp = []
        label_tmp = []
        for char, label in zip(doc.chars, doc.label):
            datas_tmp.append(lexicon[char])
            label_tmp.append(lab2idx[label])
        datas_list.append(datas_tmp)
        label_list.append(label_tmp)
        index_list.append(doc.index)

    return datas_list, label_list, index_list


def main():
    ## note
    # 把你的语料放到corpus文件夹下  我的corpus中的语料压缩了，如使用可以解压
    # 1. python embedding_model.py  -> model_conll_law.m  生成词向量文件
    # 2. python bilstm_cnn_crf.py    // is_train==1
    # 会得到 train_model.hdf5  lexicon.pkl
    # 3. 可以在之前的基础上train_model.hdf5，继续训练
    # 4. 训练完成，测试  is_train==0
    # python bilstm_cnn_crf.py  按句测试或按文件测试
    # my_weights 中存放的是我的权值

    is_train = 1  # 1/0

    if is_train == 1:
        # train  ☆☆☆☆☆☆☆
        # 训练语料路径
        corpus_path = 'corpus'
        # 初始化模型参数  可在之前的基础上训练
        base_model_weight = 'train_model.hdf5'
        nb_epoch = 1  # 迭代轮数
        process_train(corpus_path, nb_epoch, base_model_weight)

    ##############################################

    lexicon, lexicon_reverse, max_len, index_2_label = pickle.load(open('lexicon.pkl', 'rb'))
    # model
    model = Bilstm_CNN_Crf(max_len, len(lexicon), len(index_2_label), is_train=False)
    model.load_weights('train_model.hdf5')

    # 长句子测试   按标点切分后测试
    text = ''
    for i in range(10):
        text += '南京市长莅临指导，大家热烈欢迎。公交车中将禁止吃东西！'
    splitText, predLabel = word_seg_by_sentences(text, model, lexicon, max_len)
    print(splitText)

    fenci_by_file('test_documents/test_1', 'test_documents/test_1_mine', model, lexicon, max_len)


if __name__ == '__main__':
    main()
