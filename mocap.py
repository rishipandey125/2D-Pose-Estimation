import cv2
import numpy as np
#simple mocap project created by Rishi Pandey
#collect the data of the pose locations
#rework and smooth that data mathematically
#write the final videos with the smoothed data

#Path to Video File
videoPath = "test_images/serge2.mp4"
video = cv2.VideoCapture(videoPath)

def analyzeKeyPoints(video):
    # Paths for the CNN (on local machine)
    protoFile = "/Users/rishipandey125/Desktop/code/pose_estimation_model/pose_deploy_linevec_faster_4_stages.prototxt.txt"
    weightsFile = "/Users/rishipandey125/Desktop/code/pose_estimation_model/pose_iter_160000.caffemodel"

    # Reading the CNN
    network = cv2.dnn.readNetFromCaffe(protoFile,weightsFile)

    #boolean stating there is a next frame, and storing the next frame in the variable frame
    hasFrame,frame = video.read()

    #corresponding joints data for swapping
    correspondingJoints = [[2,5],[3,6],[4,7],[8,11],[9,12],[10,13]]
    count = 0
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
        hasFrame,frame = video.read()
        count += 1
        if count == 50:
            break
    return keyPoints

data = analyzeKeyPoints(video)
for x in data:
    if len(x) != 30:
        print("bad length: " + str(len(x)))
    print(x)



# skeletonPairs = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]

    #collect the data in a frame: [x cords] + [ycords] format where xcords and ycords are ALWAYS of length 15
    #then the structure will be frame num: [all cords] where all cords is on length 30 and the first 15 are x, and the last 15 are y
    #then you can smoooth each row
    #there can be no None in this list
    # draw skeleton
    # for pair in skeletonPairs:
    #     point1 = pair[0]
    #     point2 = pair[1]
    #     if keyPoints[point1] != None and keyPoints[point2] != None:
    #         #draws a line between the two corresponding points
    #         cv2.circle(img, keyPoints[point1], 20, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
    #         cv2.putText(img, "{}".format(point1), keyPoints[point1], cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    #         cv2.circle(img, keyPoints[point2], 20, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
    #         cv2.putText(img, "{}".format(point2), keyPoints[point2], cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    #         cv2.line(img, keyPoints[point1], keyPoints[point2], (0, 0, 255), 10)
    # outputVideo.write(img)
    # updating frame for next iteration
    # print("Write Frame")


# print("DONE WRITING LOOP EXITED")
# outputVideo.release()
