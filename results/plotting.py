import os
from matplotlib import pyplot as plt
import cv2

i = int(0)
images = os.listdir('u2net/images')
k = int(len(images))
for file in images:
    name = file.split('.')[0]
    
    image = cv2.imread(f'u2net/images/{name}.jpg', 0)
    groundtruth = cv2.imread(f'u2net/groundtruth/{name}.jpg', 0)
    label1 = cv2.imread(f'u2net/u2net_bce_itr_4500_train_0.811415_tar_0.056624/{name}.png', 0)
    label2 = cv2.imread(f'u2net/u2net_bce_itr_5000_train_0.745451_tar_0.047403/{name}.png', 0)
    
    plt.subplot(k,4,i*4+1),plt.imshow(image,'gray'), plt.xticks([]), plt.yticks([])
    if i == 0:
        plt.title('Images')
    plt.subplot(k,4,i*4+2),plt.imshow(groundtruth,'gray'), plt.xticks([]), plt.yticks([])
    if i == 0:
        plt.title('Ground Truth')
    plt.subplot(k,4,i*4+3),plt.imshow(label1,'gray'), plt.xticks([]), plt.yticks([])
    if i == 0:
        plt.title('U2_Net trained with 76 images')
    plt.subplot(k,4,i*4+4),plt.imshow(label2,'gray'), plt.xticks([]), plt.yticks([])
    if i == 0:
        plt.title('U2_Net trained with 94 images')
    i+=1
plt.show()