from cvzone.ClassificationModule import Classifier
import os
import cv2
import math 
from cvzone.HandTrackingModule import HandDetector
import numpy as np

VIDEO_STREAM_LINK = 'http://192.168.0.104:4747/video'
classifier = Classifier("trained_model/keras_model.h5", "trained_model/labels.txt")
CLASSES = [
    "Chandra Bindu",
    "Anusshar",
    "Bisharga",
    "Ka",
    "Kha",
    "Ga",
    "Gha",
    "Uo",
    "Ca",
    "Cha",
    "Jha",
    "Yo",
    "Ta",
    "Thha",
    "Do",
    "Dho",
    "Tha",
    "Da",
    "Dha",
    "Pa",
    "fa",
    "Ma",
    "La",
    "Ha",
    "Borgio Ja/Anta Ja",
    "Murdha Na/Donta Na",
    "Ta/Khanda Ta",
    "Ba/Bha",
    "Ba-y Ra/Da-y Ra/Dha-y Ra",
    "Talobbo sha/Danta sa/Murdha Sha"
]


def test():
    handDetector = HandDetector()
    IMG_SIZE = 300
    CROP_OFFSET = 30

    videoCaptureObject = cv2.VideoCapture(VIDEO_STREAM_LINK)

    frame_number = 0
    while True:
        success, frame = videoCaptureObject.read()
        frame_copy = frame.copy()
        if not success: 
            print(f"Failed to read frame from video stream {VIDEO_STREAM_LINK}")
        frame_number+=1
        fps = videoCaptureObject.get(cv2.CAP_PROP_FPS)
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
            prediction, index = classifier.getPrediction(imageOfHand)

            cv2.putText(frame_copy, f"{CLASSES[index]}, {prediction[index]}", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 1)
            cv2.rectangle(frame_copy, (x,y), (x+w, y+h), (255,0,255), 2)
            cv2.imshow(f"Sign Detected", frame_copy)

        if cv2.waitKey(1) == ord('q'):
            break

test()
