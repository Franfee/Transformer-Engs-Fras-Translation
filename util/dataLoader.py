# -*- coding: utf-8 -*-
# @Time    : 2022/11/28 10:37
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

import os
import sys
import pickle

from env.Env import FILE_NAME
from util.Vocab import *


def read_raw_data_nmt():
    """
    一个由Tatoeba项目的双语句子对 组成的“英－法”数据集，数据集中的每一行都是制表符分隔的文本序列对，序列对由英文文本序列和翻译后的法语文本序列组成。
    请注意，每个文本序列可以是一个句子， 也可以是包含多个句子的一个段落。
    在这个将英语翻译成法语的机器翻译问题中，英语是源语言（source language）， 法语是目标语言（target language）

    :return: 载入“英语－法语”数据集
    """
    BASE_DIR = sys.path[0]
    if "util" in BASE_DIR:
        data_dir = os.path.join(BASE_DIR, '..', 'dataset', 'fra-eng')
    else:
        data_dir = os.path.join(BASE_DIR, 'dataset', 'fra-eng')
    if not os.path.exists(data_dir):
        raise FileNotFoundError("Please check dir : %s" % data_dir)
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()


def preprocess_nmt(text):
    """
    下载数据集后，原始文本数据需要经过几个预处理步骤。
    例如，我们用空格代替不间断空格（non-breaking space）, 使用小写字母替换大写字母,并在单词和标点符号之间插入空格。

    :param text: 未经处理的原始数据
    :return: 预处理的“英语－法语”数据集
    """

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格, 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').replace('\u2009', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    return ''.join(out)


def save_processed_nmt_data(BASE_DIR, num_examples):
    # 下载并且读取数据
    print("Loading raw data...")
    raw_text = read_raw_data_nmt()
    # 预处理数据
    print("Processing raw data...")
    text = preprocess_nmt(raw_text)
    # 词元化
    print("Tokenize data...")
    source, target = tokenize_nmt(text, num_examples)
    # 转入词袋库
    print("Building Vocabulary...")
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 存储本地位置处理
    print("Storing data...")
    FILE_DIR = os.path.join(BASE_DIR, 'dataset', 'fra-eng-processed')
    FILE_LIST = [src_vocab, tgt_vocab, source, target]
    FILE_PATH = []
    for FileName in FILE_NAME:
        FILE_PATH.append(os.path.join(FILE_DIR, FileName))
    # 进行本地存储
    for File, FileName in zip(FILE_LIST, FILE_PATH):
        with open(FileName, 'wb') as _f:
            # 序列化数据
            fileObj = pickle.dumps(File)
            # 保存数据
            pickle.dump(fileObj, _f)
    print("Stored data.")


def load_processed_nmt_data(FILE_DIR, OnlyVocab=False):
    print("Loading data.")
    # 处理数据文件位置
    FILE_PATH = []
    for FileName in FILE_NAME:
        FILE_PATH.append(os.path.join(FILE_DIR, FileName))
    # 数据列表  [src_vocab, tgt_vocab, source, target]
    dataList = []
    # 读取数据
    for DataFile in FILE_PATH:
        if OnlyVocab and (len(dataList) >= 2):
            break
        with open(DataFile, 'rb') as _f:
            dataObj = pickle.load(_f)
            storedData = pickle.loads(dataObj)
            dataList.append(storedData)
    # 返回数据
    if OnlyVocab:
        return dataList[0], dataList[1]  # return src_vocab, tgt_vocab
    return dataList[0], dataList[1], dataList[2], dataList[3]   # return src_vocab, tgt_vocab, source, target


def dataLoader_nmt_data(batch_size, num_steps, num_examples=None):
    """返回翻译数据集的迭代器和词表"""
    # 处理数据文件位置
    BASE_DIR = sys.path[0]
    if "util" in BASE_DIR:
        BASE_DIR = os.path.join(BASE_DIR, '..')
    FILE_DIR = os.path.join(BASE_DIR, 'dataset', 'fra-eng-processed')

    # 不存在本地数据
    if not len(os.listdir(FILE_DIR)) > 1:
        save_processed_nmt_data(BASE_DIR, num_examples)

    # 加载数据
    src_vocab, tgt_vocab, source, target = load_processed_nmt_data(FILE_DIR)
    # 小批量化数据
    print("Building array...")
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # 生成数据迭代器
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    print(f"src_vocab.len:{len(src_vocab)}, tgt_vocab.len:{len(tgt_vocab)}")
    return data_iter, src_vocab, tgt_vocab


def test():
    # 加载器设置
    num_steps = 8
    train_iter, src_vocab, tgt_vocab = dataLoader_nmt_data(batch_size=1, num_steps=num_steps)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('batch X:', X.type(torch.int32))
        X = X.numpy().reshape(-1)
        print("batch X对应的序列:", src_vocab.to_tokens(X.tolist()))
        print('batch X的有效长度:', X_valid_len)

        print('batch Y:', Y.type(torch.int32))
        Y = Y.numpy().reshape(-1)
        print("batch Y对应的序列:", tgt_vocab.to_tokens(Y.tolist()))
        print('batch Y的有效长度:', Y_valid_len)
        break


if __name__ == '__main__':
    test()
