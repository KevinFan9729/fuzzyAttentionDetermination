import numpy as np
import cv2
from torch import imag
# img = cv.imread('test2.jpg',)
# # create a CLAHE object (Arguments are optional).
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(img)


# img = cv.imread('Unequalized_Hawkes_Bay_NZ.jpg',0)
# equ = cv.equalizeHist(img)
# res = np.hstack((img,equ)) #stacking images side-by-side

# cv.imshow('global', res)
# cv.imshow('clahe', cl1)
# cv.waitKey(10000)


# #-----Reading the image-----------------------------------------------------
# img = cv2.imread('test2.jpg', 1)
# cv2.imshow("img",img) 

# #-----Converting image to LAB Color model----------------------------------- 
# lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# cv2.imshow("lab",lab)

# #-----Splitting the LAB image to different channels-------------------------
# l, a, b = cv2.split(lab)
# cv2.imshow('l_channel', l)
# cv2.imshow('a_channel', a)
# cv2.imshow('b_channel', b)

# #-----Applying CLAHE to L-channel-------------------------------------------
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# cl = clahe.apply(l)
# cv2.imshow('CLAHE output', cl)

# #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
# limg = cv2.merge((cl,a,b))
# cv2.imshow('limg', limg)

# #-----Converting image from LAB Color model to RGB model--------------------
# final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
# cv2.imshow('final', final)
# cv2.waitKey(10000)


def preporcess(image):
    img = cv2.imread(image, 1)
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

image=preporcess('test.jpg')
cv2.imshow("test",image)

cv2.waitKey(10000)


# import cv2
# import os

# def hisEqulColor(img):
#     ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
#     channels=cv2.split(ycrcb)
#     print(len(channels))
#     cv2.equalizeHist(channels[0],channels[0])
#     cv2.merge(channels,ycrcb)
#     cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
#     return img


# fname='test2.jpg'
# img=cv2.imread(fname)

# cv2.imshow('img', img)
# img2=hisEqulColor(img)
# cv2.imshow('img2',img2)

# cv2.waitKey(10000)