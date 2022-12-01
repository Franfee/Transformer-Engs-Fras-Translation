# -*- coding: utf-8 -*-
# @Time    : 2022/11/28 10:30
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import os
import sys

from util.deviceUtil import try_gpu

# -------------------DEVICE-------------------------
DEVICE = try_gpu()

# ------------------FilePath------------------------
# project dir
BASE_DIR = sys.path[0]
if "util" in BASE_DIR:
    BASE_DIR = os.path.join(BASE_DIR, '..')
# raw source data
RAW_DIR = os.path.join(BASE_DIR, 'dataset', 'fra-eng')
RAW_FILE_NAME = "fra.txt"
# processed data file dir
FILE_DIR = os.path.join(BASE_DIR, 'dataset', 'fra-eng-processed')
# processed data file name
FILE_NAME = ["SRC_VOCAB.pkl", "TGT_VOCAB.pkl", "source.pkl", "target.pkl"]
MODEL_PATH = "model/final.mdl"

# -------------------train params------------------------
EPOCHS = 100
LR_RATE = 0.005
BATCH_SIZE = 128*4      # about 2G GPU MEM
LOSS_STOP_EPS = 0.01

# ---------------------net params------------------------
num_steps = 10
key_size, query_size, value_size = 32, 32, 32
num_hiddens, num_layers = 32, 2
norm_shape = [32]
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
drop_out = 0.1
