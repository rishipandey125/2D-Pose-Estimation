import cv2
import numpy as np
#simple mocap project created by Rishi Pandey

# Paths for the CNN (off of my git)
protoFile = "/Users/rishipandey125/Desktop/code/pose_estimation_model/pose_deploy_linevec_faster_4_stages.prototxt.txt"
weightsFile = "/Users/rishipandey125/Desktop/code/pose_estimation_model/pose_iter_160000.caffemodel"

# Reading the CNN
network = cv2.dnn.readNetFromCaffe(protoFile,weightsFile)

#Read Image
img = cv2.imread("test_images/test_rishi_pose.jpg")
imgHeight = img.shape[0]
imgWidth = img.shape[1]

#Prep Input Image for Network
inWidth = 368
inHeight = 368
inpBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
network.setInput(inpBlob)

#Output Matrix of CNN given the input Image
output = network.forward()
#height and width of output
height = output.shape[2]
width = output.shape[3]

keyPoints = []
numKeyPoints = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]
for i in range(numKeyPoints):
    confidenceMap = output[0,i,:,:]
    #only using prob and point
    minVal, prob, minLoc, point = cv2.minMaxLoc(confidenceMap)
    #KeyPoint in Threshold
    if prob > 0.1:
        #scale x and y values
        x = int((imgWidth*point[0])/width)
        y = int((imgHeight*point[1])/height)
        keyPoints.append((x,y))
        #draw a circle
        cv2.circle(img, (x,y), 20, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(img, "{}".format(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, lineType=cv2.LINE_AA)

# draw skeleton
for pair in POSE_PAIRS:
    point1 = pair[0]
    point2 = pair[1]
    if keyPoints[point1] and keyPoints[point2]:
        cv2.line(img, keyPoints[point1], keyPoints[point2], (0, 0, 255), 10)

cv2.imwrite('Output-Keypoints.jpg', img)
