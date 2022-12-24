import numpy as np
import cv2
import os

path = 'SEMSamples'
k = 0.5

def DrawPolygon(pts, image):
    if len(pts) == 0:
        pass
    else:
        pts = np.array(pts, dtype=np.uint64)
        pts = pts.reshape((-1, 1, 2))
        isClosed = True
        color = (255, 0, 0)
        thickness = 2
        image = cv2.polylines(image, [pts],
                            isClosed, color, thickness)
        return image, pts


def GetPoints(img):
    a = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x,y), radius=2, color=(0, 0, 255), thickness=-1)
            print(x, ' ', y)
            cv2.imshow('Assessing', img)
        if event==cv2.EVENT_LBUTTONDOWN:
            a.append((x,y))

    cv2.imshow('Assessing', img)
    cv2.setMouseCallback('Assessing', click_event)
    print('press "Esc" after labeling finished')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return np.array(a, dtype=np.uint64)


def showing(image, mask, polygons):
    for poly in polygons:
        image, poly = DrawPolygon(poly, image)
        mask = cv2.fillPoly(mask, [poly], color)
    cv2.imshow('image', image)
    cv2.imshow('mask', mask)


files = os.listdir(path)
images = []
for file in files:
    if file.split('.')[-1] in ['jpg', 'jpeg', 'png']:
        images.append(file)
print(images)

for imge in images:
    img = cv2.imread(f'{path}/{imge}')
    mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    image = cv2.resize(img, (0, 0), fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (0, 0), fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
    pts = []
    polygons = []
    while True:
        image_prime = image.copy()
        if len(pts) != 0:
            image_prime, pts = DrawPolygon(pts, image_prime)
        color = (255, 255, 255)
        print(f'pts: {pts}')
        if len(pts) != 0:
            polygons.append(pts)
        mask = np.zeros((image_prime.shape[0],image_prime.shape[1]), dtype=np.uint8)
        showing(image_prime, mask, polygons)
        print(f'polygons len: {len(polygons)}')
        print('press "d" for deleting the last polygon')
        if cv2.waitKey(0) == ord('d'):
            if len(polygons) != 0:
                polygons.pop()
                showing(image_prime, mask, polygons)
        print(f'polygons len: {len(polygons)}')
        mask = np.zeros((image_prime.shape[0],image_prime.shape[1]), dtype=np.uint8)
        showing(image_prime, mask, polygons)
        print('press "f" for finishing labeling current image')
        if cv2.waitKey(0) == ord('f'):
            break
        pts = GetPoints(image_prime)
    
    image = cv2.resize(image, (0, 0), fx=(1/k), fy=(1/k), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (0, 0), fx=(1/k), fy=(1/k), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f'{path}/mask_{imge}', mask)
    print('press "b" for breaking the operation')
    if cv2.waitKey(0) == ord('b'):
        break

print('Finished!')
