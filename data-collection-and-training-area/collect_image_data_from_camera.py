import os
import cv2
import math 
from cvzone.HandTrackingModule import HandDetector
import numpy as np

VIDEO_STREAM_LINK = 'http://192.168.169.203:4747/video'

def collectDataForClass(className, framesPerSecond):
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists(f'data/{className}'):
        os.mkdir(f'data/{className}')
    
    videoCaptureObject = cv2.VideoCapture(VIDEO_STREAM_LINK)

    frame_number = 0
    while True:
        success, frame = videoCaptureObject.read()
        if not success: 
            print(f"Failed to read frame from video stream {VIDEO_STREAM_LINK}")
        frame_number+=1
        fps = videoCaptureObject.get(cv2.CAP_PROP_FPS)
        if frame_number % (math.ceil(fps/framesPerSecond))==0:
            cv2.imshow(f"Capturing Video", frame)
            cv2.imwrite(f'data/{className}/{className}_{int(frame_number/(math.ceil(fps/framesPerSecond)))}.jpg', frame)
        if cv2.waitKey(1) == ord('q'):
            break

def collectHandKeypointsDataForClass(className, framesPerSecond):
    handDetector = HandDetector()
    IMG_SIZE = 300
    CROP_OFFSET = 30

    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists(f'data/{className}'):
        os.mkdir(f'data/{className}')
    
    videoCaptureObject = cv2.VideoCapture(VIDEO_STREAM_LINK)

    frame_number = 0
    while True:
        success, frame = videoCaptureObject.read()
        if not success: 
            print(f"Failed to read frame from video stream {VIDEO_STREAM_LINK}")
        frame_number+=1
        fps = videoCaptureObject.get(cv2.CAP_PROP_FPS)
        if frame_number % (math.ceil(fps/framesPerSecond))==0:
            cv2.imshow(f"Capturing Video", frame)
            hands, annotatedFrame = handDetector.findHands(frame)
            if hands:
                oneHand = hands[0]
                x,y,w,h = oneHand['bbox']
                croppedImage = annotatedFrame[y-CROP_OFFSET:y+h+CROP_OFFSET, x-CROP_OFFSET:x+w+CROP_OFFSET]
                imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8)  * 255


                aspectRatio = w / h
                if aspectRatio > 1 :
                    adjustedWidth = 300 
                    adjustedHeight = min(300, math.ceil(h*300/w))
                    croppedImage = cv2.resize(croppedImage,(adjustedWidth, adjustedHeight))

                    offset = math.ceil((IMG_SIZE-adjustedHeight)/2)
                    imgWhite[0+offset:adjustedHeight+offset, 0:adjustedWidth] = croppedImage
                else :
                    adjustedHeight = 300 
                    adjustedWidth = min(300,math.ceil(w*300/h))
                    croppedImage = cv2.resize(croppedImage,(adjustedWidth, adjustedHeight))
                    offset = math.ceil((IMG_SIZE-adjustedWidth)/2)
                    imgWhite[0:adjustedHeight, 0+offset:adjustedWidth+offset] = croppedImage 

                imageOfHand = imgWhite
                imgNum = int(frame_number/(math.ceil(fps/framesPerSecond)))
                print("Num of image saved " + str(imgNum))
                cv2.imwrite(f'data/{className}/{className}_{imgNum}.jpg', imageOfHand)
                cv2.imshow(f"Hand Detected", imageOfHand)
        if cv2.waitKey(1) == ord('q'):
            break
collectHandKeypointsDataForClass('sundor',  5)