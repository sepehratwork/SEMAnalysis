import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

images = os.listdir('SEMSamples/im_aug')
labels = os.listdir('SEMSamples/gt_aug')

i=1
for img in tqdm(images, desc='Images'):
    name = f'{i}.jpg'
    shutil.copyfile(f'SEMSamples/im_aug/{img}', f'SEMSamples/im/{name}')
    i += 1

i=1
for img in tqdm(labels, desc='Labels'):
    name = f'{i}.jpg'
    shutil.copyfile(f'SEMSamples/gt_aug/{img}', f'SEMSamples/gt/{name}')
    im = cv2.imread(f'SEMSamples/im/{name}', 1)
    gt = cv2.imread(f'SEMSamples/gt/{name}', 1)
    correct_size = cv2.resize(gt, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
    # ret,thresh1 = cv2.threshold(correct_size,50,255,cv2.THRESH_BINARY)
    # thresh1 = np.array(thresh1, dtype=np.float32)
    cv2.imwrite(f'SEMSamples/gt/{name}', correct_size)
    i+=1
