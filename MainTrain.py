# -*- coding: utf-8 -*-
# @Time    : 2022/11/27 17:09
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

# ---------------------------------import-------------------------------------
from net.Transformer import *
from util.Timer import Timer
from util.Accumulator import Accumulator
from util.dataLoader import dataLoader_nmt_data

# ---------------------------------visual-------------------------------------
from util.AnimatorVisdom import Animator
# from util.Animator import Animator

# ----------------------------------ENV---------------------------------------
from env.Env import DEVICE, MODEL_PATH
from env.Env import EPOCHS, LR_RATE, BATCH_SIZE, LOSS_STOP_EPS
from env.Env import query_size, key_size, value_size
from env.Env import ffn_num_hiddens, ffn_num_input, norm_shape, num_hiddens
from env.Env import num_heads, num_steps, num_layers, drop_out

# ----------------------------------Switch------------------------------------
TRAIN = True
TEST = True
SHOW_ATTENTION = False

# ------------------------------DATA LOADER-----------------------------------
TRAIN_ITER, SRC_VOCAB, TGT_VOCAB = dataLoader_nmt_data(BATCH_SIZE, num_steps)

# ----------------------------------------------------------------------------


def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    """
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    """
    net.to(device)
    net.train()

    lossFun = MaskedSoftmaxCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    animator = Animator(xlabel='epoch', ylabel='loss', xlim=[0, num_epochs])
    # 计时器
    timer = Timer()
    # 指标累加器 [训练损失总和, 词元数量, 批次id]
    metric = Accumulator(3)
    for epoch in range(num_epochs):
        # 指标累加器重置
        metric.reset()
        for batch in data_iter:
            if metric[2] % 10 == 0:
                print(f"epoch:{epoch + 1},batch_id:{int(metric[2])}")
            # 清空梯度
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            # 强制教学
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            Y_hat, _ = net(X, dec_input, X_valid_len)
            loss = lossFun(Y_hat, Y, Y_valid_len)
            # 损失函数的标量进行“反向传播”
            loss.sum().backward()
            # 梯度裁剪
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss.sum(), num_tokens, 1)
        # end one epoch
        if epoch % 10 == 0:
            torch.save(net.state_dict(), MODEL_PATH + "_" + "%d" % epoch)
        # visual train status
        animator.add(epoch + 1, (metric[0] / metric[1],))
        metricLoss = metric[0] / metric[1]
        print(f"epoch:{epoch + 1},metric:{(metricLoss,)}")
        # early stop
        if metricLoss < LOSS_STOP_EPS:
            print("Train Stopped early because arrived metric loss.")
            break
    # end all epochs
    print(f'loss {metric[0] / metric[1]:.3f}, {num_epochs * metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
    torch.save(net.state_dict(), MODEL_PATH)
    print("model saved")
    animator.holdFig()


if __name__ == '__main__':
    if TRAIN:
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
        print("Training net...")
        train_seq2seq(net, TRAIN_ITER, LR_RATE, EPOCHS, TGT_VOCAB, DEVICE)

    if TEST:
        from Predict import predict_seq2seq, bleu

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

        engs = ["go .", "i lost .", "he's calm .", "i'm home ."]
        fras = ["va !", "j'ai perdu .", "il est calme .", "je suis chez moi ."]
        for eng, fra in zip(engs, fras):
            translation, dec_attention_weight_seq = predict_seq2seq(
                net, eng, SRC_VOCAB, TGT_VOCAB, num_steps, DEVICE, SHOW_ATTENTION)
            print(f'{eng} => {translation}', '\t', f'bleu:{bleu(translation, fra, k=2):.3f}')

        if SHOW_ATTENTION:
            import pandas as pd
            from util.d2l import show_heatmaps, d2l

            print("Show attention:")
            # 最后一个英语到法语的句子翻译工作时, 可视化transformer的注意力权重.
            # 编码器自注意力权重的形状为（编码器层数,注意力头数,num_steps或查询的数目,num_steps或“键－值”对的数目）
            enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).\
                reshape((num_layers, num_heads, -1, num_steps))
            print("enc_attention_weights.shape:", enc_attention_weights.shape)

            # 在编码器的自注意力中,查询和键都来自相同的输入序列.
            # 因为填充词元是不携带信息的,因此通过指定输入序列的有效长度可以避免查询与使用填充词元的位置计算注意力.
            # 接下来,将逐行呈现两层多头注意力的权重.每个注意力头都根据查询、键和值的不同的表示子空间来表示不同的注意力
            show_heatmaps(enc_attention_weights.cpu(), xlabel='Key positions', ylabel='Query positions',
                          titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
            d2l.plt.show()

            # 为了可视化解码器的自注意力权重和“编码器－解码器”的注意力权重,我们需要完成更多的数据操作⼯作.
            # 例如,我们⽤零填充被掩蔽住的注意⼒权重.值得注意的是,解码器的自注意力权重和“编码器－解码器”
            # 的注意力权重都有相同的查询：即以序列开始词元（beginning-of-sequence,BOS）打头,
            # 再与后续输出的词元共同组成序列.
            dec_attention_weights_2d = [head[0].tolist()
                                        for step in dec_attention_weight_seq
                                        for attn in step for blk in attn for head in blk]
            dec_attention_weights_filled = torch.tensor(
                pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
            dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
            dec_self_attention_weights, dec_inter_attention_weights = \
                dec_attention_weights.permute(1, 2, 3, 0, 4)
            print(dec_self_attention_weights.shape, dec_inter_attention_weights.shape)
            # 由于解码器自注意力的自回归属性,查询不会对当前位置之后的“键－值”对进行注意力计算
            # Plus one to include the beginning-of-sequencetoken
            d2l.show_heatmaps(
                dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
                xlabel='Key positions', ylabel='Query positions',
                titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
            d2l.plt.show()
            # 与编码器的自注意力的情况类似,通过指定输入序列的有效长度,输出序列的查询不会与输入序列中填充位置的词元进行注意力计算
            d2l.show_heatmaps(
                dec_inter_attention_weights, xlabel='Key positions',
                ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
                figsize=(7, 3.5))
            d2l.plt.show()

"""
transformer架构是为了“序列到序列”的学习而提出的,但transformer编码器或transformer解码器通常被单独用于不同的深度学习任务中.
transformer是编码器－解码器架构的一个实践,尽管在实际情况中编码器或解码器可以单独使用.
transformer中,多头自注意力用于表示输入序列和输出序列,不过解码器必须通过掩蔽机制来保留自回归属性.
transformer中的残差连接和层规范化是训练非常深度模型的重要工具.
transformer模型中基于位置的前馈网络使用同一个多层感知机,作用是对所有序列位置的表示进行转换.
"""