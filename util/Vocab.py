# -*- coding: utf-8 -*-
# @Time    : 2022/11/28 10:38
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12


import torch
import collections
from torch.utils import data


def tokenize_nmt(text, num_examples=None):
    """
    词元化“英语－法语”数据数据集,在机器翻译中,我们更喜欢单词级词元化（最先进的模型可能使用更高级的词元化技术）.
    tokenize_nmt函数对前num_examples个文本序列对进行词元,其中每个词元要么是⼀个词,要么是⼀个标点符号.

    :param text:
    :param num_examples: 对前num_examples个文本序列对进行词元
    :return: source[i]是源语言（这里是英语）第i个文本序列的词元列表,target[i]是目标语言（这里是法语）第i个文本序列的词元列表.
    """
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i >= num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def truncate_pad(line, num_steps, padding_token):
    """语言模型中的序列样本都有⼀个固定的长度,无论这个样本是⼀个句子的⼀部分还是跨越了多个句子的⼀个片断.
    这个固定长度是由 num_steps（时间步数或词元数量）参数指定的.
    在机器翻译中,每个样本都是由源和目标组成的文本序列对,其中的每个文本序列可能具有不同的长度.
    为了提高计算效率,我们仍然可以通过截断（truncation）和 填充（padding）方式实现⼀次只处理⼀个小批量的文本序列.
    假设同⼀个小批量中的每个序列都应该具有相同的长度num_steps,那么如果文本序列的词元数目少于num_steps时,我们将继续在其末尾添加特定的“<pad>”词元,直到其长度达到num_steps；
    反之,我们将截断文本序列时,只取其前num_steps 个词元,并且丢弃剩余的词元.
    这样,每个文本序列将具有相同的长度,以便以相同形状的小批量进行加载.
    """
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[line] for line in lines]
    lines = [line + [vocab['<eos>']] for line in lines]
    array = torch.tensor([truncate_pad(line, num_steps, vocab['<pad>']) for line in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def count_corpus(tokens_in):
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens_in) == 0 or isinstance(tokens_in[0], list):
        # 将词元列表展平成⼀个列表
        tokens_in = [token for line in tokens_in for token in line]
    return collections.Counter(tokens_in)


class Vocab:
    # 词元的类型是字符串,⽽模型需要的输⼊是数字,因此这种类型不方便模型使用.
    # 构建⼀个字典,通常也叫做词表（vocabulary）,用来将字符串类型的词元映射到从0开始的数字索引中.
    # 将训练集中的所有文档合并在⼀起,对它们的唯⼀词元进行统计,得到的统计结果称之为语料（corpus）.
    # 然后根据每个唯⼀词元的出现频率,为其分配⼀个数字索引.很少出现的词元通常被移除,这可以降低复杂性.
    # 另外,语料库中不存在或已删除的任何词元都将映射到⼀个特定的未知词元“ < unk >”.我们可以选择增加⼀个
    # 列表,用于保存那些被保留的词元,例如：填充词元（“ < pad >”）；序列开始词元（“ < bos >”）；序列结束词元（“ < eos >”）.
    """文本词表"""

    def __init__(self, tokens_in=None, min_freq=0, reserved_tokens=None):
        if tokens_in is None:
            tokens_in = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens_in)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens_in):
        if not isinstance(tokens_in, (list, tuple)):
            return self.token_to_idx.get(tokens_in, self.unk)
        return [self.__getitem__(token) for token in tokens_in]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs
