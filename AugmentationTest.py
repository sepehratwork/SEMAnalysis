import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A

def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

image = cv2.imread('SEMSamples/im_aug/20.jpg')
mask = cv2.imread('SEMSamples/gt_aug/20.jpg')

original_height, original_width = image.shape[:2]

aug = A.PadIfNeeded(min_height=128, min_width=128, p=1)
aug1 = A.RandomBrightnessContrast(p=1,brightness_limit=(-0.5,0.5), contrast_limit=0.5)
aug2 = A.GaussianBlur(p=1, blur_limit=(5, 9), sigma_limit=1)
aug3 = A.HorizontalFlip(p=1)
aug4 = A.VerticalFlip(p=1)
aug5 = A.Rotate(limit=180,interpolation=cv2.INTER_LINEAR,p=1)
aug6 = A.Transpose(p=1)
aug7 = A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03,
                            border_mode=cv2.BORDER_REFLECT101, mask_value=None, same_dxdy=False)
aug8 = A.GridDistortion(p=1, num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR, 
                            border_mode=cv2.BORDER_REFLECT101)
aug9 = A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1, interpolation=cv2.INTER_LINEAR, 
                            border_mode=cv2.BORDER_REFLECT101)
aug10 = A.RandomSizedCrop(min_max_height=(500, 1001), height=original_height, width=original_width, p=1, 
                            interpolation=cv2.INTER_LINEAR)
aug11 = A.GaussNoise(p=1, var_limit=(10,50), mean=0.1)


augmented = aug(image=image, mask=mask)
img = augmented['image']
mask = augmented['mask']
cv2.imwrite('test/Image.jpg', img)
cv2.imwrite('test/Mask.jpg', mask)

augmented = aug1(image=image, mask=mask)
img1 = augmented['image']
mask1 = augmented['mask']
cv2.imwrite('test/Image1.jpg', img1)
cv2.imwrite('test/Mask1.jpg', mask1)

augmented = aug2(image=image, mask=mask)
img2 = augmented['image']
mask2 = augmented['mask']
cv2.imwrite('test/Image2.jpg', img2)
cv2.imwrite('test/Mask2.jpg', mask2)

augmented = aug3(image=image, mask=mask)
img3 = augmented['image']
mask3 = augmented['mask']
cv2.imwrite('test/Image3.jpg', img3)
cv2.imwrite('test/Mask3.jpg', mask3)

augmented = aug4(image=image, mask=mask)
img4 = augmented['image']
mask4 = augmented['mask']
cv2.imwrite('test/Image4.jpg', img4)
cv2.imwrite('test/Mask4.jpg', mask4)

augmented = aug5(image=image, mask=mask)
img5 = augmented['image']
mask5 = augmented['mask']
cv2.imwrite('test/Image5.jpg', img5)
cv2.imwrite('test/Mask5.jpg', mask5)

augmented = aug6(image=image, mask=mask)
img6 = augmented['image']
mask6 = augmented['mask']
cv2.imwrite('test/Image6.jpg', img6)
cv2.imwrite('test/Mask6.jpg', mask6)

augmented = aug7(image=image, mask=mask)
img7 = augmented['image']
mask7 = augmented['mask']
cv2.imwrite('test/Image7.jpg', img7)
cv2.imwrite('test/Mask7.jpg', mask7)

augmented = aug8(image=image, mask=mask)
img8 = augmented['image']
mask8 = augmented['mask']
cv2.imwrite('test/Image8.jpg', img8)
cv2.imwrite('test/Mask8.jpg', mask8)

augmented = aug9(image=image, mask=mask)
img9 = augmented['image']
mask9 = augmented['mask']
cv2.imwrite('test/Image9.jpg', img9)
cv2.imwrite('test/Mask9.jpg', mask9)

augmented = aug10(image=image, mask=mask)
img10 = augmented['image']
mask10 = augmented['mask']
cv2.imwrite('test/Image10.jpg', img10)
cv2.imwrite('test/Mask10.jpg', mask10)

augmented = aug11(image=image, mask=mask)
img11 = augmented['image']
mask11 = augmented['mask']
cv2.imwrite('test/Image11.jpg', img11)
cv2.imwrite('test/Mask11.jpg', mask11)