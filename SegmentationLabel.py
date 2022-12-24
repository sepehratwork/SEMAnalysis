import numpy as np
import cv2
import os
import DrawPolygon
import GetPoints

list = ['jpg','png']
path = 'G:\MachineLearning\yasin\Segmentation'
list_main = os.listdir(path)
list_prime = []
for file in list_main:
    if file.split('.')[-1] in list:
        list_prime.append(file)

counter = 0
for img in list_prime:
    
    if cv2.waitKey(0) == ord('q'):
        break

    counter += 1
    image = cv2.imread(path + f'\{img}')
    mask = np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8)
    k = 2
    image = cv2.resize(image, (0, 0), fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (0, 0), fx=k, fy=k, interpolation=cv2.INTER_LINEAR)

    while True:
        print('Select Your Mask')
        pts = GetPoints.GetPoints(image)        
        
        print('drawing image')
        image, pts = DrawPolygon.DrawPolygon(pts, image)
        
        color = (255, 255, 255)

        mask = cv2.fillPoly(mask, [pts], color)
        
        print('press "n" for going to next image')
        if cv2.waitKey(0) == ord('n'):
            break
        
        while True:
            print('press "Esc" to finish showing mask')
            print('press "p" for easing the last point of polygon')

            if cv2.waitKey(0) == ord('p'):
                print('Just Popped a Point ...')
                pts.pop()
                image, pts = DrawPolygon.DrawPolygon(pts, image)
                color = (255, 255, 255)
                mask = cv2.fillPoly(mask, [pts], color)

            if cv2.waitKey(0) == ord('b'):
                break

            cv2.imshow('Mask', mask)
            if cv2.waitKey(0):
                break
    
    image = cv2.resize(image, (0, 0), fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (0, 0), fx=k, fy=k, interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(f'Mask{counter}.png', mask)
