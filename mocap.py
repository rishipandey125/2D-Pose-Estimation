import cv2
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
although we have like 18 points it will be out of 44 because past the first 18 is a heatmap
third dimension is the height of the output map (where it is +y?)
fourth dimension is the width of the output map (where it is +x?)
'''
for x in output[0]:
    print(x)
