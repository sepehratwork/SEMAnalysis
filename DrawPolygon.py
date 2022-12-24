import cv2
import numpy as np
import GetPoints


def DrawPolygon(pts, image):
    # Polygon corner points coordinates
    # pts = GetPoints.GetPoints(image)

    pts = np.array(pts)

    pts = pts.reshape((-1, 1, 2))

    isClosed = True

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.polylines() method
    # Draw a Blue polygon with
    # thickness of 1 px
    image = cv2.polylines(image, [pts],
                        isClosed, color, thickness)


    return image, pts


# # for testing
# # path
# path = '20210809_081648.jpg'

# # Reading an image in default
# # mode
# image = cv2.imread(path)

# image, pts = DrawPolygon(image)

# # Displaying the image
# while(1):
#     cv2.imshow('image', image)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
        
# cv2.destroyAllWindows()