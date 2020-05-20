import cv2
import numpy as np
import ROI

image = 'test_image.jpg'
image_colored = cv2.imread('test_image.jpg')
image_gray = cv2.imread(image, 0)

coordinates, ROI = ROI.get_roi(image_gray)

for item in coordinates:
    cv2.rectangle(image_colored, (item[0], item[2]), (item[1], item[3]), (0, 0, 0), 1)

cv2.imshow('result', image_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()