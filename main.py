from mocap import *
import cv2
import numpy as np
videoPath = "/Users/rishipandey125/Desktop/testVideosMOCAP/test2.mp4"

points = [(540, 626), (493, 793), (446, 793), (375, 876), (422, 960), (540, 793), (563, 876), (610, 918), (446, 960), (493, 1126), (446, 1252), (516, 960), (586, 1126), (540, 1252), (493, 876)]

img = np.zeros((1500,1500,3), np.uint8)
for point in points:
    print(str(point))
    cv2.circle(img, point, 20, (255, 255, 255), thickness=-1, lineType=cv2.FILLED)
    cv2.putText(img,str(point),point, cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, lineType=cv2.LINE_AA)

cv2.imshow('image', img)
cv2.waitKey(0)
