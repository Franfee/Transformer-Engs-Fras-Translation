# -*- coding: utf-8 -*-
# @Time    : 2022/11/27 16:03
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12


from .TransformerComponent import *


class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addNorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addNorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addNorm1(X, self.attention(X, X, X, valid_lens))
        return self.addNorm2(Y, self.ffn(Y))


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False):
        """
        transformer编码器
        """
        super(TransformerEncoder, self).__init__()
        self.attention_weights = None
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size,
                                              num_hiddens, norm_shape,
                                              ffn_num_input, ffn_num_hiddens, num_heads,
                                              dropout, use_bias))

    def forward(self, X, valid_lens):
        # 因为位置编码值在-1和1之间,因此嵌入值乘以嵌入维度的平方根进行缩放,然后再与位置编码相加.
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout, i):
        """
        解码器的第i块
        """
        super(DecoderBlock, self).__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addNorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addNorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addNorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段,输出序列的所有词元都在同一时间处理, 因此state[2][self.i]初始化为None.
        # 预测阶段,输出序列是通过词元一个接着一个解码的, 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addNorm1(X, X2)
        # 编码器－解码器注意力.
        # enc_outputs的开头: (batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addNorm2(Y, Y2)
        return self.addNorm3(Z, self.ffn(Z)), state


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout):
        super(TransformerDecoder, self).__init__()
        self._attention_weights = None
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 DecoderBlock(key_size, query_size, value_size,
                                              num_hiddens, norm_shape,
                                              ffn_num_input, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


def ComponentTest():
    # ==============check point 0================
    # 经过掩蔽softmax操作,超出有效长度的值都被掩蔽为0
    mask_value1 = masked_softmax(torch.rand(2, 2, 8), valid_lens=torch.tensor([2, 3]))
    # 同样,我们也可以使用⼆维张量,为矩阵样本中的每⼀行指定有效长度。
    mask_value2 = masked_softmax(torch.rand(2, 2, 8), valid_lens=torch.tensor([[1, 3], [2, 4]]))
    print("测试mask:", mask_value1)
    print("测试mask:", mask_value2)

    # 使用键和值相同的小例⼦来测试我们编写的MultiHeadAttention类。
    # 多头注意力输出的形状是  (batch_size,num_queries,num_hiddens)
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
    attention.eval()
    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    print("MultiHeadAttention output shape:", attention(X, Y, Y, valid_lens).shape)

    # ==============check point 1================
    ffn = PositionWiseFFN(4, 4, 8)
    ffn.eval()
    print("前馈网络块测试:")
    print(ffn(torch.ones((2, 3, 4)))[0])

    # ==============check point 2================
    add_norm = AddNorm([3, 4], 0.5)
    add_norm.eval()
    print("Add & LayerNorm shape:", add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)

    # ==============check point 3================
    X = torch.ones((2, 100, 24))
    valid_lens = torch.tensor([3, 2])
    encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    encoder_blk.eval()
    print("编码器块 shape:", encoder_blk(X, valid_lens).shape)

    # ==============check point 4================
    # 下面我们指定了超参数来创建一个两层的transformer编码器. Transformer编码器输出的形状是（批量大小,时间步数目,num_hiddens）.
    encoder = TransformerEncoder(200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
    encoder.eval()
    print("transformer编码器 shape:", encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)

    # ==============check point 5================
    # 为了便于在“编码器－解码器”注意力中进行缩放点积计算和残差连接中进行加法计算,编码器和解码器的特征维度都是num_hiddens.
    decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
    decoder_blk.eval()
    X = torch.ones((2, 100, 24))
    state = [encoder_blk(X, valid_lens), valid_lens, [None]]
    print("解码器块 shape:", decoder_blk(X, state)[0].shape)

    # ==============check point 6================
    # 现在我们构建了由num_layers个DecoderBlock实例组成的完整的transformer解码器.
    # 最后,通过一个全连接层计算所有vocab_size个可能的输出词元的预测值.解码器的自注意力权重和编码器解码器注意力权重都被存储下来,方便日后可视化的需要.
    decoder = TransformerDecoder(200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
    decoder.eval()
    net = EncoderDecoder(encoder, decoder)
    print(net)


if __name__ == '__main__':
    ComponentTest()
