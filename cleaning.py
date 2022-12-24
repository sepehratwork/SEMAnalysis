import os
from tqdm import tqdm
import cv2
import numpy as np

labels = os.listdir('SEMSamples/gt_aug')
images = os.listdir('SEMSamples/im_aug')

i = 1
for image in tqdm(images, desc='images'):
    os.replace(f'SEMSamples/im_aug/{image}',f'SEMSamples/im_aug/{i}.jpg')
    i += 1
    # if len(image.split('_')) > 3:
        # os.remove(f'SEMSamples/im_aug/{image}')

i = 1
for lbl in tqdm(labels, desc='labels'):
    os.replace(f'SEMSamples/gt_aug/{lbl}',f'SEMSamples/gt_aug/{i}.jpg')
    name = f'{i}.jpg'
    im = cv2.imread(f'SEMSamples/im_aug/{name}', 1)
    gt = cv2.imread(f'SEMSamples/gt_aug/{name}', 1)
    correct_size = cv2.resize(gt, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
    # ret,thresh1 = cv2.threshold(correct_size,50,255,cv2.THRESH_BINARY)
    # thresh1 = np.array(thresh1, dtype=np.float32)
    cv2.imwrite(f'SEMSamples/gt_aug/{name}', correct_size)
    i += 1
    # if len(lbl.split('_')) > 3:
        # os.remove(f'SEMSamples/gt_aug/{lbl}')