import os
import cv2
import numpy as np
from tqdm import tqdm
import math
# from matplotlib import pyplot as plt


# ground_truth = os.listdir('test/groundtruth')
# predicated = os.listdir('test/predicted')

# predicate = cv.imread('test/predicted/67.png', 0)
# ret,thresh1 = cv.threshold(predicate,127,255,cv.THRESH_BINARY)
# ret,thresh2 = cv.threshold(predicate,80,255,cv.THRESH_BINARY)
# gt = cv.imread('test/groundtruth/67.jpg', 0)
# difference = gt - thresh2

# plt.subplot(2,2,1),plt.imshow(thresh1,'gray',vmin=0,vmax=255)
# plt.title('thresh1')
# plt.subplot(2,2,2),plt.imshow(thresh2,'gray',vmin=0,vmax=255)
# plt.title('thresh2')
# plt.subplot(2,2,3),plt.imshow(gt,'gray',vmin=0,vmax=255)
# plt.title('ground truth')
# plt.subplot(2,2,4),plt.imshow(difference,'gray',vmin=0,vmax=255)
# plt.title('difference')
# plt.show()

# from keras import backend as K
# import cv2
# import numpy as np

# def iou_coef(y_true, y_pred, smooth=1):
#   intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
#   union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
#   iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
#   return iou

# def dice_coef(y_true, y_pred, smooth=1):
#   intersection = K.sum(y_true * y_pred, axis=[1,2,3])
#   union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
#   dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
#   return dice

# y_true = cv2.imread('test/groundtruth/67.jpg', cv2.IMREAD_GRAYSCALE)
# y_pred = cv2.imread('test/predicted/67.png', cv2.IMREAD_GRAYSCALE)

# while True:
#   cv2.imshow('1', y_pred)
#   if cv2.waitKey(0):
#     break

# ret,y_true = cv2.threshold(y_true,50,255,cv2.THRESH_BINARY)
# ret,y_pred = cv2.threshold(y_pred,50,255,cv2.THRESH_BINARY)

# while True:
#   cv2.imshow('2', y_pred)
#   if cv2.waitKey(0):
#     break

# y_true = np.array(y_true, dtype=np.float32)
# y_pred = np.array(y_pred, dtype=np.float32)

# while True:
#   cv2.imshow('3', y_pred)
#   if cv2.waitKey(0):
#     break

# fault = y_true - y_pred

# while True:
#   cv2.imshow('fault', fault)
#   if cv2.waitKey(0):
#     break

# print(y_pred)
# print(y_pred.ravel()[-10:-1])

# import cv2
# import numpy as np

# img = cv2.imread('test/76.png', cv2.IMREAD_GRAYSCALE)
# ret,thresh = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
# print((img[0][0]))
# print((thresh[0][0]))
# fault = 0
# for i in range(img.shape[1]):
#   for j in range(img.shape[0]):
    # if (img[i][j] != 0) and (img[i][j] != 255):
      # pass
      # fault += 1
# print(fault)

x = (0,0)
y = (1,0)
z = (0,1)

x = np.array(x)
y = np.array(y)
z = np.array(z)
pi = math.pi

a = y - x
b = z - y

theta = np.arccos((np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)))
theta = 180 - ((theta / pi) * 180)

print(theta)
