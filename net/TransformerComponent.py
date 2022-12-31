# -*- coding: utf-8 -*-
# @Time    : 2022/11/27 16:21
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

import torch
import math
from torch import nn


def transpose_qkv(X, num_heads):
    """
    为了多注意力头的并行计算而变换形状
    输入X的形状:(batch_size,查询或者“键－值”对的个数,num_hiddens)
    """
    # X的形状:(batch_size,查询或者“键－值”对的个数,num_heads, num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # X的形状:(batch_size,num_heads,查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


def sequence_mask(X, valid_len, value=0.):
    """
    在每个时间步,解码器预测了输出词元的概率分布。应该将填充词元的预测排除在损失函数的计算之外。
    为此,我们可以使用下面的sequence_mask函数通过零值化屏蔽不相关的项,以便后面任何不相关预测的计算都是与零的乘积,结果都等于零。
    例如,如果两个序列的有效长度（不包括填充词元）分别为1和2,则第⼀个序列的第⼀项和第⼆个序列的前两项之后的剩余项将被清除为零。

    """
    seq_len = X.size(1)
    valid_len = valid_len.cuda()
    mask = torch.arange(seq_len).cuda()[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量,valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换,从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """
    [cuda]上计算的的带遮蔽的softmax交叉熵损失函数, 可以通过扩展softmax交叉熵损失函数来遮蔽不相关的预测。
    最初,所有预测词元的掩码都设置为1。⼀旦给定了有效长度,与填充词元对应的掩码将被设置为0。
    最后,将所有词元的损失乘以掩码,以过滤掉损失中填充词元产生的不相关预测。

    """
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)

    # CELoss(交叉熵损失)等价predict经log_softmax后执行nn.NLLLoss(negative log likelihood loss,负对数似然损失)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len).float()
        # weights = weights.cuda()
        # 输出N个loss_i的列表,默认为"mean",还有"sum"可选
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss.cuda() * weights.cuda()).mean(dim=1)
        return weighted_loss


class AdditiveAttention(nn.Module):
    """
    加性注意力
    ⼀般来说,当查询和键是不同长度的矢量时,我们可以使用加性注意力作为评分函数。
    给定查询q∈R^k和键k∈ R^k,加性注意力（additive attention）的评分函数为:
            a(q, k) = w_v.T * tanh(Wq*Q + Wk*K) ∈ R
    其中可学习的参数是Wq ∈ R^(h×q)、Wk ∈ R^(h×k)和 wv ∈ R^h。
    将查询和键连结起来后输入到⼀个多层感知机（MLP）中,感知机包含⼀个隐藏层,其隐藏单元数是⼀个超参数h。
    通过使用tanh作为激活函数,并且禁用偏置项。

    """

    def __init__(self, key_size, query_size, num_hiddens, dropout):
        super(AdditiveAttention, self).__init__()
        self.attention_weights = None
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后,
        # queries的形状：(batch_size,查询的个数,1,num_hidden)  key的形状：(batch_size,1,“键－值”对的个数,num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出,因此从形状中移除最后那个维度。
        # scores的形状：(batch_size,查询的个数,“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size,“键－值”对的个数,值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """
    缩放点积注意力
    使用点积可以得到计算效率更高的评分函数,但是点积操作要求查询和键具有相同的长度d。
    假设查询和键的所有元素都是独立的随机变量,并且都满足零均值和单位方差,那么两个向量的点积的均值为0,方差为d。
    为确保无论向量长度如何,点积的方差在不考虑向量长度的情况下仍然是1,我们将点积除以√d,
    则缩放点积注意力（scaled dot-product attention）评分函数为：
                   a(q, k) = q.⊤ * K / √d.
    在实践中,我们通常从小批量的⻆度来考虑提高效率,例如基于n个查询和m个键－值对计算注意力,其中查询和键的长度为d,值的长度为v。
    查询Q ∈ R^(n×d)、键K∈ R^(m×d)和值V ∈ R^(m×v)的缩放点积注意力是：
                   softmax(QK.⊤ / √d)*V   ∈ R^(n×v)
    在下面的缩放点积注意力的实现中使用Dropout进行模型正则化。

    """
    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.attention_weights = None
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size,查询的个数,d)
    # keys的形状：(batch_size,“键－值”对的个数,d)
    # values的形状：(batch_size,“键－值”对的个数,值的维度)
    # valid_lens的形状:(batch_size,)或者(batch_size,查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries,keys,values的形状: (batch_size,查询或者“键－值”对的个数,num_hiddens)
        # valid_lens　的形状: (batch_size,)或(batch_size,查询的个数)

        # 经过变换后,输出的queries,keys,values　的形状: (batch_size*num_heads,查询或者“键－值”对的个数,num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0,将第一项（标量或者矢量）复制num_heads次, 然后如此复制第二项,然后诸如此类。
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads,查询的个数, num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size,查询的个数,num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) \
            / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        """
        基于位置的前馈网络(MLP层)
        改变张量的最里层维度的尺寸,会改变成基于位置的前馈网络的输出尺寸.
        因为用同一个多层感知机对所有位置上的输入进行变换,所以当所有这些位置的输入相同时,它们的输出也是相同的.

        :param ffn_num_input:
        :param ffn_num_hiddens:
        :param ffn_num_outputs:
        """
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout):
        """
        Add&Norm层, 残差连接后进行层规范化
        在一个小批量的样本内基于批量规范化对数据进行重新中心化和重新缩放的调整.
        LN(层规范化)和BN(批量规范化)的目标相同:将数据样本变换为均值0方差1.
        但LN是基于特征维度进行规范化.BN是样本批次纬度进行规范化.
        尽管BN(批量规范化)在计算机视觉中被广泛应用,但在自然语言处理任务中（输入通常是变长序列）批量规范化通常不如层规范化的效果好.

        :param normalized_shape:
        :param dropout:
        """
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        # 残差连接要求两个输入的形状相同,以便加法操作后输出张量的形状相同.
        return self.ln(self.dropout(Y) + X)

