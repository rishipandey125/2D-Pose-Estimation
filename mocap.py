import cv2
import numpy as np
#simple mocap project created by Rishi Pandey

'''
Current Improvements: Joint Swapping, No Overlap, Right Connects (None in spots not detected in KeyPoints)
Savgol Smoothing?
'''
# Paths for the CNN (off of my git)
protoFile = "/Users/rishipandey125/Desktop/code/pose_estimation_model/pose_deploy_linevec_faster_4_stages.prototxt.txt"
weightsFile = "/Users/rishipandey125/Desktop/code/pose_estimation_model/pose_iter_160000.caffemodel"

# Reading the CNN
network = cv2.dnn.readNetFromCaffe(protoFile,weightsFile)

#Path to Video File
videoPath = "test_images/serge.mp4"
video = cv2.VideoCapture(videoPath)
#boolean stating there is a next frame, and storing the next frame in the variable frame
hasFrame,frame = video.read()
#write output video
outputVideo = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]

#loop through the below with each video frame, write that drawn skeleton to the outputVideo
while hasFrame:
    img = frame
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

    #keyPoints is a list of the locations of the 15 points, main data needed for writing to animation
    keyPoints = []
    numKeyPoints = 15
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
        else:
            keyPoints.append(None)

    # draw skeleton
    for pair in POSE_PAIRS:
        point1 = pair[0]
        point2 = pair[1]
        try:
            #The reason we are getting index out of bounds, is because there are supposed to be
            #15 points, but sometimes it is only recognizing 12 joints so keyPoints size is only 12
            #Therefore when trying to "connect the dots", we have dots trying to be connected that don't exist
            if keyPoints[point1]!= None and keyPoints[point2] != None:
                #draws a line between the two corresponding points
                cv2.line(img, keyPoints[point1], keyPoints[point2], (0, 0, 255), 10)
        except IndexError:
            print("Index Out of Bounds on Frame " + str(count))
    outputVideo.write(img)
    # updating frame for next iteration
    print("Write Frame")
    hasFrame,frame = video.read()


print("DONE WRITING LOOP EXITED")
outputVideo.release()
