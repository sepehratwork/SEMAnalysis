import cv2
import numpy as np
import math

def angles(cnt):
    cnt = np.array(cnt)
    print(cnt)
    if len(cnt) == 0:
        return 0
    # calculate each angle of the poligon
    pi = math.pi
    angle = np.zeros(len(cnt))
    for i in range(len(cnt)):
        if i == 0:
            x = cnt[len(cnt)-1][0]
            y = cnt[i][0]
            z = cnt[i+1][0]
        elif i == (len(cnt)-1):
            x = cnt[i-1][0]
            y = cnt[i][0]
            z = cnt[0][0]
        else:
            x = cnt[i-1][0]
            y = cnt[i][0]
            z = cnt[i+1][0]
        a = [0,0]
        b = [0,0]
        a[0] = y[0] - x[0]
        a[1] = y[1] - x[1]
        b[0] = z[0] - y[0]
        b[1] = z[1] - y[1]
        a = [0, 1]
        b = [-4, 4]
        theta = np.arccos((np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)))
        theta = 180 - ((theta / pi) * 180)
        angle[i] = theta
    return angle

# x = [0,0]
# y = [3,0]
# z = [0,4]

# pi = math.pi
# a = [0,0]
# b = [0,0]
# a[0] = y[0] - x[0]
# a[1] = y[1] - x[1]
# b[0] = z[0] - y[0]
# b[1] = z[1] - y[1]
# print(f'a: {a}, b: {b}')
# theta = np.arccos((np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)))
# theta = 180 - ((theta / pi) * 180)
# print(theta)


cnt = [[[0,0]],[[1,0]],[[1,1]],[[0,8]]]
angle = angles(cnt)
print(angle)






# Test
# img = cv2.imread('test/92.png')
# ret,img = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
# contours,hierarchy = cv2.findContours(img, 1, 2)
# for i in range(len(contours)):
#     img2 = img.copy()
#     cnt = contours[i]
#     hull = cv2.convexHull(cnt)
#     epsilon = 0.00000000001*cv2.arcLength(hull,True)
#     approx = cv2.approxPolyDP(cnt,epsilon,True)
#     cv2.drawContours(img2, [approx], 0, (50, 50, 50), 2) 
#     img3 = img.copy()
#     angle = angles(approx)
#     for j in range(len(cnt)):
#         point = cnt[j][0]
#         radius = 1
#         color = (100,100,100)
#         thickness = 1
#         img3 = cv2.circle(img3, point, radius, color, thickness)
#         x = point[0]
#         y = point[1]
#         cv2.putText(img3,f"{angle[j]}", (x,y), cv2.FONT_HERSHEY_PLAIN, 1, 100)
#     cv2.imshow('Angles', img3)