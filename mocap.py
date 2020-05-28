import cv2
import numpy as np
#simple mocap project created by Rishi Pandey

#return true if two joints are relatively in the same location (detecting the same point)
#false otherwise
def overlappingJoints(point1,point2,minDist):
    distance = np.sqrt(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1]))
    #we need a new comparison distance
    if distance <= minDist:
        return True
    else:
        return False

'''
Current Improvements: Joint Swapping, No Overlap, Missing Joint Estimation
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

skeletonPairs = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]
correspondingJoints = [[2,5],[3,6],[4,7],[8,11],[9,12],[10,13]]

#loop through the below with each video frame, write that drawn skeleton to the outputVideo
count = 0
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
    corresponding_index = 0
    for i in range(numKeyPoints):
        confidenceMap = output[0,i,:,:]
        #only using prob and point
        minVal, prob, minLoc, point = cv2.minMaxLoc(confidenceMap)
        #KeyPoint in Threshold
        if prob > 0.1:
            #check 5,6,7 and 11,12,13 for joint swapping
            #scale x and y values
            x = int((imgWidth*point[0])/width)
            y = int((imgHeight*point[1])/height)
            keyPoints.append((x,y))
            #index out of bounds here
            if corresponding_index < len(correspondingJoints):
                if i == correspondingJoints[corresponding_index][1]:
                    previousPoint = correspondingJoints[corresponding_index][0]
                    corresponding_index += 1
                    if keyPoints[previousPoint] != None:
                        #check for overlap
                        if overlappingJoints(keyPoints[previousPoint],keyPoints[i],imgWidth/50):
                            print("Detected/Fixing Overlap")
                            #which one is correct?
                            if keyPoints[i][0] >= (imgWidth/2):
                                # the right keypoint is correct
                                keyPoints[previousPoint] = None
                            else:
                                #the left keypoint is correct
                                keyPoints[i] = None
                        else: #check for swap
                            if keyPoints[previousPoint][0] > keyPoints[i][0]:
                                print("Detected/Fixing Swap")
                                #swap
                                tempPoint = keyPoints[i]
                                keyPoints[i] = keyPoints[previousPoint]
                                keyPoints[previousPoint] = tempPoint
        else:
            keyPoints.append(None)

    # draw skeleton
    for pair in skeletonPairs:
        point1 = pair[0]
        point2 = pair[1]
        if keyPoints[point1] != None and keyPoints[point2] != None:
            #draws a line between the two corresponding points
            cv2.circle(img, keyPoints[point1], 20, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(img, "{}".format(point1), keyPoints[point1], cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.circle(img, keyPoints[point2], 20, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(img, "{}".format(point2), keyPoints[point2], cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.line(img, keyPoints[point1], keyPoints[point2], (0, 0, 255), 10)
    outputVideo.write(img)
    # updating frame for next iteration
    print("Write Frame")
    count += 1
    if count == 50:
        break
    hasFrame,frame = video.read()


print("DONE WRITING LOOP EXITED")
outputVideo.release()
