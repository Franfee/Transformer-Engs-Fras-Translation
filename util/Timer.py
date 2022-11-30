# -*- coding: utf-8 -*-
# @Time    : 2022/10/24 21:33
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.7.9


import time
import numpy as np


class Timer: 
    """记录多次运行时间"""
    def __init__(self):
        self.tik = None
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


def test_fun():
    print("starting tik...")
    timer = Timer()
    time.sleep(2)
    print("ended tok...")
    print(timer.stop())


if __name__ == '__main__':
    test_fun()



