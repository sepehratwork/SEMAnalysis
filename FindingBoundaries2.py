# finding contours

# Moments
import numpy as np
import cv2 as cv
import os
import math


def angles(cnt):
    cnt = np.array(cnt)
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
        print(f'a: {a}, b:{b}')
        theta = np.arccos((np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)))
        theta = 180 - ((theta / pi) * 180)
        angle[i] = theta
    return angle
    


path = 'SEMSamples/gt/'

for image in os.listdir(path):
    org = cv.imread('SEMSamples_Copy/1.jpg',0)
    img = cv.imread('SEMSamples_Copy/mask_1.jpg',0)
    n = 1
    img = cv.resize(img, (0, 0), fx=n, fy=n, interpolation=cv.INTER_LINEAR)
    ret,img = cv.threshold(img,100,255,cv.THRESH_BINARY)
    cv.imshow('image', img)
    cv.waitKey(0)
    contours,hierarchy = cv.findContours(img, 1, 2)
    print(f'{len(contours)} Contours')
    if cv.waitKey(0) == ord('b'):
        break

    for i in range(len(contours)):
        print('\n')
        print(f'contour {i+1}')
        cnt = contours[i]
        M = cv.moments(cnt)
        # print( M )
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print(f'center: {(cx, cy)}')
        # Contour Area
        area = cv.contourArea(cnt)
        if area > 100:
            print(len(cnt))
            print(cnt.shape)
            print(f'area: {area}')
            
            org2 = org.copy()
            img2 = img.copy()
            
            # Contour Perimeter
            perimeter = cv.arcLength(cnt,True)
            print(f'perimeter: {perimeter}')
            
            # Convex Hull
            hull = cv.convexHull(cnt)
            # cv.drawContours(img2,[hull],0,(150,150,150),5)

            # Contour Approximation
            epsilon = 0.00000000001*cv.arcLength(hull,True)
            # approx = cv.approxPolyDP(hull,epsilon,True)
            approx = cv.approxPolyDP(cnt,epsilon,True)
            cv.drawContours(img2, [approx], 0, (50, 50, 50), 2)
            cv.drawContours(org2, [approx], 0, (50, 50, 50), 2)

            # Checking Convexity
            k = cv.isContourConvex(cnt)
            print(f'Convexity: {k}')

            # Showing angles
            angle = angles(approx)
            # print(angle)
            img3 = img.copy()
            for j in range(len(cnt)):
                point = cnt[j][0]
                radius = 1
                color = (100,100,100)
                thickness = 1
                img3 = cv.circle(img3, point, radius, color, thickness)
                x = point[0]
                y = point[1]
                cv.putText(img3,f"{angle[j]}", (x,y), cv.FONT_HERSHEY_PLAIN, 1, 100)
            cv.imshow('Angles', img3)
            cv.waitKey(0)

            # # Straight Bounding Rectangle
            # x,y,w,h = cv.boundingRect(cnt)
            # cv.rectangle(img2,(x,y),(x+w,y+h),(100,100,100),2)

            # # Rotated Rectangle
            # rect = cv.minAreaRect(cnt)
            # box = cv.boxPoints(rect)
            # box = np.int0(box)
            # cv.drawContours(img2,[box],0,(150,150,150),2)

            # # Minimum Enclosing Circle
            # (x,y),radius = cv.minEnclosingCircle(cnt)
            # center = (int(x),int(y))
            # radius = int(radius)
            # cv.circle(img2,center,radius,(200,200,200),2)

            # # Fitting an Ellipse
            # ellipse = cv.fitEllipse(cnt)
            # cv.ellipse(img2,ellipse,(250,250,250),2)

            # # Fitting a Line
            # rows,cols = img2.shape[:2]
            # [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
            # lefty = int((-x*vy/vx) + y)
            # righty = int(((cols-x)*vy/vx)+y)
            # cv.line(img2,(cols-1,righty),(0,lefty),(0,0,0),2)

            cv.imshow(f'approxPolyDP contour', img2)
            cv.waitKey(0)
            cv.imshow(f'approxPolyDP contour', org2)
            cv.waitKey(0)
            cv.destroyAllWindows()
