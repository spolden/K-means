import numpy
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

lst = os.listdir('murphy\\test')
is_Vertical = False  # 是否垂直

for item in lst:
    pth = os.path.join('murphy\\test', item)  # 遍历每个文件

    with open(pth, 'r') as file:
        mat = np.zeros((60, 60))
        raw = file.readlines()  # 读取原始数据

        for i in range(0, len(raw)):
            raw[i] = raw[i].strip()  # 去除\n

        for data in raw:
            data = data.split()  # 分割数据集

            x, y = int(data[3]), int(data[4])  # 求画向量所需参数
            u, v = int(data[5]), int(data[6])
            cv2.line(mat, (x, y), (u, v), (225, 225, 225))
    cv2.imshow('test', np.flip(mat, 0))
    cv2.waitKey(0)
