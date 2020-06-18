import cv2
import numpy as np
import pandas as pd
from scipy import signal
#WEEKEND OF JUNE 12th:
# ALWAYS MAKE FIRST AND LAST FRAMES SUPER SIMPLE AND CORRECT
    #Take Correct Videos and Test (5 Segments)
    #3D point estimation
#simple mocap project created by Rishi Pandey
#collect the data of the pose locations
#rework and smooth that data mathematically
#write the final videos with the smoothed data


#idea for 3d: take the head and split it into x,y,z and then smooth and reassemble
#tomasi kanade algo seems to be something worthwhile
#Path to Video File


'''
Anaylyze Keypoints Function
Uses Video Input to Return List of Keypoints for Each Frame
'''
def analyzeKeyPoints(path):
    # Paths for the CNN (on local machine)
    protoFile = "/Users/rishipandey125/Desktop/code/pose_estimation_model/pose_deploy_linevec_faster_4_stages.prototxt.txt"
    weightsFile = "/Users/rishipandey125/Desktop/code/pose_estimation_model/pose_iter_160000.caffemodel"

    # Reading the CNN
    network = cv2.dnn.readNetFromCaffe(protoFile,weightsFile)

    baseVideo = cv2.VideoCapture(path)
    #boolean stating there is a next frame, and storing the next frame in the variable frame
    hasFrame,frame = baseVideo.read()

    #corresponding joints data for swapping
    correspondingJoints = [[2,5],[3,6],[4,7],[8,11],[9,12],[10,13]]
    keyPoints = []
    while hasFrame:
        imgHeight, imgWidth = frame.shape[0], frame.shape[1]

        #Prep Input Image for Network
        inWidth, inHeight = 368, 368
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        network.setInput(inpBlob)

        #Output Matrix of CNN given the input Image
        output = network.forward()

        #height and width of output
        height,width = output.shape[2], output.shape[3]

        numKeyPoints = 15
        corresponding_index = 0
        x_keyPoints, y_keyPoints = [], []
        for i in range(numKeyPoints):
            confidenceMap = output[0,i,:,:]
            #only using prob and point
            minVal, prob, minLoc, point = cv2.minMaxLoc(confidenceMap)
            #KeyPoint in Threshold or just grab data for the first frame
            if prob > 0.1 or len(keyPoints) == 0:
                #scale x and y values
                x = int((imgWidth*point[0])/width)
                y = int((imgHeight*point[1])/height)
                x_keyPoints.append(x)
                y_keyPoints.append(y)
                #index in bounds to check for swap?
                if corresponding_index < len(correspondingJoints):
                    #at the right index?
                    if i == correspondingJoints[corresponding_index][1]:
                        previousPoint = correspondingJoints[corresponding_index][0]
                        corresponding_index += 1
                        if x_keyPoints[previousPoint] > x_keyPoints[i]: #swap!
                            tempPoint_x, tempPoint_y = x_keyPoints[i], y_keyPoints[i]
                            x_keyPoints[i], y_keyPoints[i] = x_keyPoints[previousPoint], x_keyPoints[previousPoint]
                            x_keyPoints[previousPoint], y_keyPoints[previousPoint] = tempPoint_x, tempPoint_y
            else:
                x_keyPoints.append(previous_x[i])
                y_keyPoints.append(previous_y[i])
        # updating frame for next iteration
        previous_x, previous_y = x_keyPoints, y_keyPoints
        keyPoints.append(x_keyPoints + y_keyPoints)
        hasFrame,frame = baseVideo.read()
    return keyPoints

'''
Smooth Data Helper Function using Savgol Smoothing
'''
def smoothData(data):
    df = pd.DataFrame.from_records(data)
    # smooth data!
    #add the third dimension and make each frame a list of 45 in length then smooth the same way
    for x in range(len(data[0])):
        #window_length = 15 and polyorder = 2
        df[x] = signal.savgol_filter(df[x], 15, 2)
    return df

'''
Motion Capture Function
Uses Video Path to Generate Simple Motion Capture Data and Draws a Skeleton Over the Tracked Person
'''
def motionCapture(path):
    #video object from path
    video = cv2.VideoCapture(path)
    hasFrame,frame = video.read()
    #match this to input framrate
    outputFrameRate = 24
    outputVideo = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), outputFrameRate, (frame.shape[1],frame.shape[0]))
    #pairs of points to be aconnected to create a skeleton
    skeletonPairs = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]
    data = analyzeKeyPoints(path)
    #data smoothed
    df = smoothData(data)
    frameCount = 0
    while hasFrame and frameCount < len(df[0]):
        for pair in skeletonPairs:
            point1 = pair[0]
            point2 = pair[1]
            cord1 = tuple(np.array([df[point1][frameCount],df[point1+15][frameCount]],int))
            cord2 = tuple(np.array([df[point2][frameCount],df[point2+15][frameCount]],int))

            #draws skeleton on given frame
            cv2.circle(frame, cord1, 20, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(point1), cord1, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, cord2, 20, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(point2), cord2, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.line(frame, cord1, cord2, (0, 0, 255), 10)
        outputVideo.write(frame)
        frameCount += 1
        hasFrame,frame = video.read()
    outputVideo.release()


def poseData(path):
    data = analyzeKeyPoints(path)
    df = smoothData(data)
    numKeyPoints = 15
    pose2D = []
    for frameIndex in range(len(df)):
        frameList = []
        for x in range(numKeyPoints):
            frameList.append([df[x][frameIndex],df[x+15][frameIndex]])
        pose2D.append(frameList)
    return pose2D

videoPath = "/Users/rishipandey125/Desktop/testVideosMOCAP/test2.mp4"
print(poseData(videoPath))
