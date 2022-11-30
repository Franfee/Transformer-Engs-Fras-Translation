# -*- coding: utf-8 -*-
# @Time    : 2022/11/28 15:36
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

import numpy as np
import matplotlib.pyplot as plt


class Animator:
    def __init__(self, xlabel=None, ylabel=None, xlim=None, ylim=None, title=None, legend=None, suptitle=None):
        plt.figure(0)
        # 开启一个画图的窗口进入交互模式，用于实时更新数据
        plt.ion()
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止中文标签乱码
        plt.rcParams['axes.unicode_minus'] = False
        self.x = []
        self.y = []
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.title = title
        self.legend = legend
        if isinstance(xlabel, list):
            self.fig_num = len(xlabel)
        else:
            self.fig_num = 1
        self.suptitle = suptitle

    def add(self, x, y):
        # 清除刷新前的图表，防止数据量过大消耗内存
        plt.clf()
        # 打开网格
        plt.grid()
        # 存在总标题,并设置文字大小
        if self.suptitle:
            plt.suptitle(self.suptitle, fontsize=30)

        # 存在子图
        if self.fig_num > 1:
            raise "TO DO Err: realizations to be doing..."
        else:
            # 只有一幅图
            self.x.append(x)
            self.y.append(y)
            plt.plot(self.x, self.y)
            if self.xlabel:
                plt.xlabel(self.xlabel)
            if self.ylabel:
                plt.ylabel(self.ylabel)
            if self.xlim:
                plt.xlim(self.xlim)
            if self.ylim:
                plt.ylim(self.ylim)
            if self.title:
                plt.title(self.title)
            if self.legend:
                plt.legend(self.legend)
        # 设置暂停时间，太快图表无法正常显示
        plt.pause(0.5)

    def holdFig(self):
        self.x = None
        self.y = None
        plt.ioff()  # 关闭交互模式
        plt.show()  # 显示图片,防止闪退


if __name__ == '__main__':
    ani = Animator(xlabel='epoch', ylabel='loss', xlim=[0, 200])
    for i in range(200):
        ani.add(i, np.abs(10.0 * np.random.randn()))
    ani.holdFig()
