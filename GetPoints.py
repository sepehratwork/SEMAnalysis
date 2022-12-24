# importing the module
import cv2
import numpy as np


def GetPoints(img):
    a = []

    # function to display the coordinates of
    # of the points clicked on the image
    def click_event(event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x,y), radius=2, color=(0, 0, 255), thickness=-1)
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)
            # displaying the coordinates
            # on the image window
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img, str(x) + ',' +
            #             str(y), (x,y), font,
            #             1, (255, 0, 0), 1)
            cv2.imshow('Assessing', img)
            # cv2.circle(img, (x,y), radius=2, color=(255, 0, 0), thickness=-1)
        
        if event==cv2.EVENT_LBUTTONDOWN:
            a.append((x,y))
        
        # if event==cv2.EVENT_FLAG_ALTKEY:
        #     a.pop()
        
    
    # reading the image

    # displaying the image
    cv2.imshow('Assessing', img)

    # setting mouse hadler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('Assessing', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()
    print(a)

    return a

# # for testing
# img = cv2.imread("20210809_081648.jpg")
# GetPoints(img)
