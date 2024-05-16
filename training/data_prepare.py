# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:18:18 2019

@author: Administrator
"""

import numpy as np
from scipy import io as sio
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import cv2
import os
import csv
import pandas as pd


if __name__ == '__main__':
    fname = 'datalist_gopro_train.txt'
    nms_train = []
    with open(fname, 'r+', encoding='utf-8') as f:
        for i in f.readlines():
            nms_train.append(i.split(' ')[0])
            nms_train.append(i.split(' ')[1][:-1])

    fname = 'datalist_gopro_test.txt'
    nms_test = []
    with open(fname, 'r+', encoding='utf-8') as f:
        for i in f.readlines():
            nms_test.append(i.split(' ')[0])
            nms_test.append(i.split(' ')[1][:-1])

    im_size = 280
    imgs_train = np.zeros((len(nms_train), 3, im_size, im_size), dtype=np.uint8)
    imgs_train2 = np.zeros((len(nms_train), 3, im_size // 2, im_size // 2), dtype=np.uint8)
    imgs_train3 = np.zeros((len(nms_train), 3, im_size // 4, im_size // 4), dtype=np.uint8)
    for i in np.arange(0, len(nms_train)):
        im = cv2.cvtColor(cv2.imread('./' + nms_train[i], 1), cv2.COLOR_BGR2RGB)
        # plt.imshow(im)
        # plt.show()
        imgs_train[i] = cv2.resize(im, (im_size, im_size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        imgs_train2[i] = cv2.resize(im, (im_size // 2, im_size // 2), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        imgs_train3[i] = cv2.resize(im, (im_size // 4, im_size // 4), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)

    im_size = 256
    imgs_test = np.zeros((len(nms_test), 3, im_size, im_size), dtype=np.uint8)
    imgs_test2 = np.zeros((len(nms_test), 3, im_size // 2, im_size // 2), dtype=np.uint8)
    imgs_test3 = np.zeros((len(nms_test), 3, im_size // 4, im_size // 4), dtype=np.uint8)
    for i in np.arange(0, len(nms_test)):
        im = cv2.cvtColor(cv2.imread('./' + nms_test[i], 1), cv2.COLOR_BGR2RGB)
        # plt.imshow(im)
        # plt.show()
        imgs_test[i] = cv2.resize(im, (im_size, im_size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        imgs_test2[i] = cv2.resize(im, (im_size // 2, im_size // 2), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        imgs_test3[i] = cv2.resize(im, (im_size // 4, im_size // 4), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)

    sio.savemat('Gopro_Area_280_140_70.mat',
                {'X': imgs_train[1::2], 'Y': imgs_train[::2], 'X2': imgs_train2[1::2], 'Y2': imgs_train2[::2],
                 'X3': imgs_train3[1::2], 'Y3': imgs_train3[::2], 'names_train': nms_train[1::2],
                 'Xtest': imgs_test[1::2], 'Ytest': imgs_test[::2], 'Xtest2': imgs_test2[1::2],
                 'Ytest2': imgs_test2[::2], 'Xtest3': imgs_test3[1::2], 'Ytest3': imgs_test3[::2],
                 'names_test': nms_test[1::2]})




