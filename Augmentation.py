import os
import cv2
import albumentations as A
from tqdm import tqdm


aug1 = A.RandomBrightnessContrast(p=1,brightness_limit=(-0.3,0.3), contrast_limit=0.5)
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
aug11 = A.GaussNoise(p=1, var_limit=(10,50), mean=0.1)



labels = os.listdir('SEMSamples/gt_aug')
images = os.listdir('SEMSamples/im_aug')

try:
    for img in tqdm(images, desc='Augmenting Images ...'):
        # reading original image
        image = cv2.imread(f'SEMSamples/im_aug/{img}')
        mask = cv2.imread(f'SEMSamples/gt_aug/{img}')
        original_height, original_width = image.shape[:2]
        name = img.split('.')[0]

        # augmentation
        #1
        augmented = aug1(image=image, mask=mask)
        image_RandomBrightnessContrast = augmented['image']
        mask_RandomBrightnessContrast = augmented['mask']
        cv2.imwrite(f'SEMSamples/im_aug/{name}_1.jpg', image_RandomBrightnessContrast)
        cv2.imwrite(f'SEMSamples/gt_aug/{name}_1.jpg', mask_RandomBrightnessContrast)

        #2
        augmented = aug2(image=image, mask=mask)
        image_GaussianBlur = augmented['image']
        mask_GaussianBlur = augmented['mask']
        cv2.imwrite(f'SEMSamples/im_aug/{name}_2.jpg', image_GaussianBlur)
        cv2.imwrite(f'SEMSamples/gt_aug/{name}_2.jpg', mask_GaussianBlur)

        #3
        augmented = aug3(image=image, mask=mask)
        image_h_flipped = augmented['image']
        mask_h_flipped = augmented['mask']
        cv2.imwrite(f'SEMSamples/im_aug/{name}_3.jpg', image_h_flipped)
        cv2.imwrite(f'SEMSamples/gt_aug/{name}_3.jpg', mask_h_flipped)

        #4
        augmented = aug4(image=image, mask=mask)
        image_v_flipped = augmented['image']
        mask_v_flipped = augmented['mask']
        cv2.imwrite(f'SEMSamples/im_aug/{name}_4.jpg', image_v_flipped)
        cv2.imwrite(f'SEMSamples/gt_aug/{name}_4.jpg', mask_v_flipped)

        #5
        augmented = aug5(image=image, mask=mask)
        image_rot90 = augmented['image']
        mask_rot90 = augmented['mask']
        cv2.imwrite(f'SEMSamples/im_aug/{name}_5.jpg', image_rot90)
        cv2.imwrite(f'SEMSamples/gt_aug/{name}_5.jpg', mask_rot90)

        #6
        augmented = aug6(image=image, mask=mask)
        image_transposed = augmented['image']
        mask_transposed = augmented['mask']
        cv2.imwrite(f'SEMSamples/im_aug/{name}_6.jpg', image_transposed)
        cv2.imwrite(f'SEMSamples/gt_aug/{name}_6.jpg', mask_transposed)

        #7
        augmented = aug7(image=image, mask=mask)
        image_ElasticTransform = augmented['image']
        mask_ElasticTransform = augmented['mask']
        cv2.imwrite(f'SEMSamples/im_aug/{name}_7.jpg', image_ElasticTransform)
        cv2.imwrite(f'SEMSamples/gt_aug/{name}_7.jpg', mask_ElasticTransform)

        #8
        augmented = aug8(image=image, mask=mask)
        image_GridDistortion = augmented['image']
        mask_GridDistortion = augmented['mask']
        cv2.imwrite(f'SEMSamples/im_aug/{name}_8.jpg', image_GridDistortion)
        cv2.imwrite(f'SEMSamples/gt_aug/{name}_8.jpg', mask_GridDistortion)

        #9
        augmented = aug9(image=image, mask=mask)
        image_OpticalDistortion = augmented['image']
        mask_OpticalDistortion = augmented['mask']
        cv2.imwrite(f'SEMSamples/im_aug/{name}_9.jpg', image_OpticalDistortion)
        cv2.imwrite(f'SEMSamples/gt_aug/{name}_9.jpg', mask_OpticalDistortion)

        # #10
        # aug10 = A.RandomSizedCrop(min_max_height=(500, 1000), height=original_height, width=original_width, p=1, 
        #                             interpolation=cv2.INTER_LINEAR)
        # augmented = aug10(image=image, mask=mask)
        # image_RandomSizedCrop = augmented['image']
        # mask_RandomSizedCrop = augmented['mask']
        # cv2.imwrite(f'SEMSamples/im_aug/{name}_10.jpg', image_RandomSizedCrop)
        # cv2.imwrite(f'SEMSamples/gt_aug/{name}_10.jpg', mask_RandomSizedCrop)

        #11
        augmented = aug11(image=image, mask=mask)
        image_GaussNoise = augmented['image']
        mask_GaussNoise = augmented['mask']
        cv2.imwrite(f'SEMSamples/im_aug/{name}_11.jpg', image_GaussNoise)
        cv2.imwrite(f'SEMSamples/gt_aug/{name}_11.jpg', mask_GaussNoise)

except:
    pass