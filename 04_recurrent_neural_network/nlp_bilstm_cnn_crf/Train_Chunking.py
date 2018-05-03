#!/usr/bin/python
# -*- coding:utf-8 -*-

# This script trains the BiLSTM-CNN-CRF architecture for Chunking in English using
# the CoNLL 2000 dataset (https://www.clips.uantwerpen.be/conll2000/chunking/).
# The code use the embeddings by Komninos et al. (https://www.cs.york.ac.uk/nlp/extvec/)


import os
import sys
import logging
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import prepareDataset, loadDatasetPickle


# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)
os.chdir(dirname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


############################################################################################################
#
# 1.Data preprocessing
#
############################################################################################################
datasets = {
        'conll2000_chunking':                                   # Name of the dataset
        {'columns': {0: 'tokens', 1: 'POS', 2: 'chunk_BIO'},    # 0:tokens,1:POS,2:chunk information using BIO encoding
         'label': 'chunk_BIO',                                  # Which column we like to predict
         'evaluate': True,                                      # Set true always for single task setups
         'commentSymbol': None}                                 # Lines starting with this string will be skipped
}

# :: Path on your computer to the word embeddings. Embeddings by Komninos et al. will be downloaded automatically ::
# :: 词向量文件地址,采样Komninos词向量;没有则自动下载
embeddingsPath = 'komninos_english_embeddings.gz'

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
# :: 数据预处理,并保存为cPickle文件
pickleFile = prepareDataset(embeddingsPath, datasets)


############################################################################################################
#
# 2.Network training
#
############################################################################################################
# :: Load the embeddings and the dataset ::
# :: 加载词向量和训练数据 ::
embeddings, mappings, data = loadDatasetPickle(pickleFile)
params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25)}


model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.storeResults('results/conll2000_chunking.csv')    # Path to store performance scores for dev / test
model.modelSavePath = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
model.fit(epochs=5)



