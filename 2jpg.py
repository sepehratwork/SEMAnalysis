import os

images = os.listdir('SEMSamples/gt_aug')

for image in images:
  name = image.split('.')[0]
  os.rename(f'SEMSamples/gt_aug/{image}',
            f'SEMSamples/gt_aug/{name}.jpg')

images = os.listdir('SEMSamples/im_aug')

for image in images:
  name = image.split('.')[0]
  os.rename(f'SEMSamples/im_aug/{image}',
            f'SEMSamples/im_aug/{name}.jpg')
