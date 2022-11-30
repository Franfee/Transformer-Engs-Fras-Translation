# -*- coding: utf-8 -*-
# @Time    : 2022/11/29 18:55
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import torch
from util.dataLoader import load_processed_nmt_data
from net.Transformer import EncoderDecoder, TransformerDecoder, TransformerEncoder
from Predict import predict_seq2seq

# ------------------net param-------------------
from env.Env import *
# ----------------------------------------------

# ------------------------------DATA LOADER-----------------------------------
SRC_VOCAB, TGT_VOCAB = load_processed_nmt_data(FILE_DIR, True)
# ----------------------------------------------------------------------------


def GenNet():
    print("Establishing encoder...")
    encoder = TransformerEncoder(len(SRC_VOCAB), key_size, query_size, value_size,
                                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                                 num_heads, num_layers, drop_out)
    print("Establishing decoder...")
    decoder = TransformerDecoder(len(TGT_VOCAB), key_size, query_size, value_size,
                                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                                 num_heads, num_layers, drop_out)
    print("Establishing net model...")
    net = EncoderDecoder(encoder, decoder)
    print("Loading model parameters...")
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return net


if __name__ == '__main__':
    example = ["go .", "i lost .", "he's calm .", "i'm home ."]

    cloneNet = GenNet()
    print("example:")
    for eng in example:
        translation, dec_attention_weight_seq = predict_seq2seq(
            cloneNet, eng, SRC_VOCAB, TGT_VOCAB, num_steps, DEVICE, False)
        print(f'{eng} => {translation}')

    while True:
        print("=================================================================")
        inputs = str(input("Input English which need to be translate into France\n->"))
        translation, dec_attention_weight_seq = predict_seq2seq(
            cloneNet, inputs, SRC_VOCAB, TGT_VOCAB, num_steps, DEVICE, False)
        print(f'{inputs} => {translation}')
