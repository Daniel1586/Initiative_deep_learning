#!/usr/bin/python
# -*- coding:utf-8 -*-

# This script trains the BiLSTM-CRF architecture for part-of-speech tagging
# using the universal dependency dataset (http://universaldependencies.org/).
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
        'unidep_pos':                           # Name of the dataset
        {'columns': {1: 'tokens', 3: 'POS'},    # Column 1 contains tokens, column 3 contains POS information
         'label': 'POS',                        # Which column we like to predict
         'evaluate': True,                      # Set true always for single task setups
         'commentSymbol': None}                 # Lines in the input data starting with this string will be skipped
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
params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25)}

model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.storeResults('results/unidep_pos_results.csv')    # Path to store performance scores for dev/test
model.modelSavePath = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"    # Path to store models
model.fit(epochs=5)
