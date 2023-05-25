# -*- coding: UTF-8 -*-
"""
@Project ：DeepLearning 
@File    ：Convert8to24.py
@Author  ：Zdy
@Date    ：2023/5/15 16:42 
"""
import multiprocessing
import cv2
import numpy as np
import os
imgpath = r"H:\DeepLearning\unet-pytorch\VOCdevkit\VOC2007\SegmentationClass1"
savepath = r"H:\DeepLearning\unet-pytorch\VOCdevkit\VOC2007\SegmentationClass"
def convert8224(pngpath):
    img = np.zeros((960,1280,3),np.uint8)

    src = cv2.imread(os.path.join(imgpath,pngpath),0)
    #print(src.height)
    #cv2.imshow('src', src)
    #cv2.waitKey()
    #print(src.shape)

    ret1,needle = cv2.threshold(src,200,255,cv2.THRESH_BINARY)
    ret2,board = cv2.threshold(src, 200, 255, cv2.THRESH_TOZERO_INV)
    #ret3,board1 = cv2.threshold(board, 100, 255, cv2.THRESH_BINARY)


    #cv2.imshow('needle',needle)

    #cv2.imshow('board', board1)

    for h in range(src.shape[0]):
        for w in range (src.shape[1]):
            if needle[h,w] == 255:
                img[h,w]=255
            if board[h,w] > 100:
                img[h,w] = 127
    #cv2.imshow('img', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(savepath,pngpath),img)
    #cv2.imwrite(os.path.join(savepath,"_"+pngpath),board)
    #cv2.waitKey()
threads = []
if __name__ == "__main__":
    # list_img = os.listdir(imgpath)
    # # 多进程
    # p = multiprocessing.Pool(8)  # 起8个进程
    # p.map(convert8224, list_img)
    # p.close()
    # p.join()
    path = os.path.join(imgpath,"label12_26_20_50_53c11_j2_0.png") #imgpath + "label12_26_20_50_53c11_j2_0.png";
    convert8224("label12_26_20_50_53c11_j2_0.png")
