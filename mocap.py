import cv2
import numpy as np
#simple mocap project created by Rishi Pandey
#first goal is to identify pose and draw skeletal mesh on img

# Paths for the CNN (off of my git)
protoFile = "/Users/rishipandey125/Desktop/code/pose_estimation_model/pose_deploy_linevec_faster_4_stages.prototxt.txt"
weightsFile = "/Users/rishipandey125/Desktop/code/pose_estimation_model/pose_iter_160000.caffemodel"

# Reading the CNN
network = cv2.dnn.readNetFromCaffe(protoFile,weightsFile)

#Read Image
img = cv2.imread("image.jpeg")

imgHeight = img.shape[0]
imgWidth = img.shape[1]

#Prep Input Image for Network
inWidth = 368
inHeight = 368
inpBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
network.setInput(inpBlob)

#Output Matrix of CNN given the input Image
output = network.forward()
#working on parsing the output

'''
first dimension is the image id

second dimension is the index of the keypoint (which number point is it)
and its confidence map
although we have like 15 points it will be out of 44 because past the first 15 is a heatmap
What does the confidence map mean? The confidence map can be accessed as output[0,x,:,:]
The confidence map for every keypoint x where (x in numKeyPoints) is the "heat map" around that keypointself.
The max in that confidence map is the specific point on that heatmap that is the most precise point.
For example if we are identifying the knee cap. A heat map will be drawn around the general location.
We want the most precise location so we would find the global maxima of that heatmap. Think of it as a plotting of a 3d
function over a joint. You want to find that maxima and its location to know the precise location of the knee cap.

third dimension is the height of the output map (where it is +y?)
fourth dimension is the width of the output map (where it is +x?)
'''

keyPoints = []
numKeyPoints = 15

for x in range(numKeyPoints):
    confidenceMap = output[0,x,:,:]
    print(confidenceMap) #this produces a confidence map
    print("---------")
