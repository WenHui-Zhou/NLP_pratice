import numpy as np 
import pandas as pd 
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
# The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and 
# statistical natural language processing for English written in the Python programming language.
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from bs4 import BeautifulSoup
import re

#TQDM is a progress bar library with good support for nested loops and Jupyter/IPython notebooks.
from tqdm import tqdm

"""
输入为pd数据，df = pd.read_csv(file.csv,sep='\t')
输出为tokenizer后的结果
"""
def clean_sentences(df):
    reviews = []
    for sent in tqdm(df['Phrase']):
        review_text = BeautifulSoup(sent).get_text() # 去除html
        review_text = re.sub('[^a-zA-Z]',' ',review_text) # 去除非文本部分
        # tokenize the sentences
        words = word_tokenize(review_text.lower()) # 令牌化
        lemma_words = [lemmatizer.lemmatize(i) for i in words] # 词形恢复
        reviews.append(lemma_words)
        return reviews

"""
将数据转为one-hot,输入为list
"""
from keras.utils import to_categorical
def to_onehot(target):
    y_target = to_categorical(target)
    num_classes = y_target.shape[1]
    return y_target

"""
划分训练集
"""
def split_dataset(trainset,y_target):
    from sklearn.model_selection import train_test_split
    x_train,x_val,y_train,y_val = train.test_split(trainset,y_target,test_size = 0.2,stratify = y_target)
    return x_train,x_val,y_train,y_val

#使用keras搭建网络
from keras.utils import to_categorical # 转成one-hot
import random
from tensorflow import set_random_seed
from sklearn.model_selection import train_test_split # 划分验证集，训练集
from keras.preprocessing import sequence # 将词转成序列
from keras.preprocessing.text import Tokenizer # 来自keras的token函数
from keras.layers import Dense,Dropout,Embedding,LSTM # 网络层
from keras.callbacks import EarlyStopping # 训练提早结束
from keras.losses import categorical_crossentropy # 分类器
from keras.optimizers import Adam # 优化器
from keras.models import Sequential # 模型容器

"""
对句子进行序列化
"""
def sentences_to_sequence(x_train,x_val,x_test):
    # 首先获得总体的词汇
    # 去除重复出现的词，unique_words里头是一个单词的全集
    unique_words = set()
    len_max = 0
    for sent in tqdm(x_train):
        unique_words.update(sent)
        if(len_max < len(sent)):
            len_max = len(sent)
    # 对句子进行tokenizer操作
    tokenizer = Tokenizer(num_words = len(list(unique_words)))
    tokenizer.fit_on_texts(list(x_train)) # 用数据初始化tokenizer
    # tokenizer.word_count 返回一个字典，字典的key为词，val为出现的个数
    # tokenizer.word_index 对词集合中每一个词编号,key为词，val为编号
    # 将句子中的词替换成词的编号
    x_train = tokenizer.texts_to_sequences(x_train)
    x_val = tokenizer.texts_to_sequences(x_val)
    x_test = tokenizer.texts_to_sequences(test_sentences)
    # 由于每个句子的长度不一样长，因此需要对齐，通过pad在短的句子开头补上0
    x_train = sequence.pad_sequences(x_train,maxlen=len_max)
    x_val = sequence.pad_sequences(x_val,maxlen=len_max)
    x_test = sequence.pad_sequences(x_test,maxlen=len_max)

"""
keras 搭建一个网络
"""

def build_model():
    # 设置early stop
    early_stopping = EarlyStopping(min_delta=0.001,mode ='max',monitor='val_acc',patience = 2)
    callback = [early_stopping]
    # keras搭建模型
    model = Sequential()
    # embedding(input_dim(词汇表长度),output_dim(输出的vector的长度)，input_length(输入句子的长度))
    # 等于输入了词汇表，句子的sequences，然后去学习word2vec的参数，得到表示句子的vector，长度通常设置成128或300
    model.add(Embedding(len(list(unique_words)),300,input_length=len_max))# embedding 起到word2vec的作用
    # LSTM
    model.add(LSTM(128,dropout=0.5,recurrent_dropout=0.5,return_sequences=True))
    model.add(LSTM(64,dropout=0.5,recurrent_dropout=0.5,return_sequences=False))
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',optimizer=Adam(lr = 0.005),metrics=['accuracy'])
    model.summary()
    return model

def train_model(x_val,y_val,x_train,y_train,x_test):
    # 往模型中加入数据,开始训练
    history = model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=6,batch_size = 256,verbose=1,callbacks=callback)
    import matplotlib.pyplot as plt
    %matplotlib inline
    epoch_count = range(1,len(history.history['loss']) + 1)
    # 绘制出训练结果
    plt.plot(epoch_count,history.history['loss'],'r--')
    plt.plot(epoch_count,history.history['val_loss'],'b--')
    plt.legend(['Training loss','validation loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    # 模型预测
    y_pred = model.predict_classes(x_test)
    return y_pred
