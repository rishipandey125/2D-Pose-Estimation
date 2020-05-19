import cv2
#simple mocap project created by Rishi Pandey
#first goal is to identify pose and draw skeletal mesh on img

# Paths for the CNN (off of my git)
#it cant read the prototxt file
protoFile = "Desktop/code/pose_estimation_model/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "Desktop/code/pose_estimation_model/pose_iter_160000.caffemodel"

# Reading the CNN
net = cv2.dnn.readNetFromCaffe(protoFile,weightsFile)
