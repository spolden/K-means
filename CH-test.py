import numpy
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt


def read_data(option):
    labels = []
    lst = os.listdir('murphy\\' + option)
    length = len(lst)
    mats = np.zeros((length, 3600))  # 初始化训练用数组
    itr = 0
    for item in lst:
        pth = os.path.join('murphy\\' + option, item)  # 遍历每个文件

        with open(pth, 'r') as file:
            mat = np.zeros((60, 60))
            raw = file.readlines()  # 读取原始数据

            for i in range(0, len(raw)):
                raw[i] = raw[i].strip()  # 去除\n

            for data in raw:
                data = data.split()  # 分割数据集

                x, y = int(data[3]), int(data[4])  # 求画线所需参数
                u, v = int(data[5]), int(data[6])
                cv2.line(mat, (x, y), (u, v), (1, 1, 1))  # 转化为矩阵

        result = np.flip(mat, 0).reshape((1, 3600))  # 矩阵转化为1*3600向量
        labels.append(item[0].upper())  # 标签
        mats[itr, :] = result  # 向训练用数组内导入数据
        itr = itr + 1  # 换行

    return labels, mats
