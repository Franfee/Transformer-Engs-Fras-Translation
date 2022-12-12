# -*- coding: utf-8 -*-
# @Time    : 2022/11/29 21:15
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12
import time

import numpy as np
from visdom import Visdom


class Animator:
    def __init__(self, xlabel=None, ylabel=None, xlim=None, ylim=None, title=None, legend=None, suptitle='title'):
        self.viz = Visdom()
        self.opts = dict()
        self.opts['xtickmin'] = xlim[0]
        self.opts['xtickmax'] = xlim[1]

        if xlabel:
            self.opts['xlabel'] = xlabel
        if ylabel:
            self.opts['ylabel'] = ylabel
        if xlim:
            self.opts['xlim'] = xlim
        if ylim:
            self.opts['ylim'] = ylim
        if legend:
            self.opts['legend'] = legend
        if title:
            self.opts['title'] = title
        self.suptitle = suptitle
        # Y的第一个点的坐标,X的第一个点的坐标,窗口的名称,图像的标例
        if (self.opts['legend'] is None) or len(self.opts['legend']) == 1:
            self.viz.line([0.], [0.], win=self.suptitle, opts=self.opts)
        else:
            start_y = []
            for _ in self.opts['legend']:
                start_y.append(0.)
            self.viz.line([start_y], [0.], win=self.suptitle, opts=self.opts)

    def add(self, x, y):
        self.viz.line([y], [x], win=self.suptitle, update='append')

    def holdFig(self):
        # 兼容Animator代码跑空
        pass


if __name__ == '__main__':
    ani = Animator(xlabel='epoch', ylabel='loss', xlim=[0, 200])
    for i in range(200):
        ani.add(i, np.abs(10.0 * np.random.randn()))
        time.sleep(0.1)
    ani.holdFig()
