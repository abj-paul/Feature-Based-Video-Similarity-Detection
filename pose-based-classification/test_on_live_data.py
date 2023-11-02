import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os
import sys
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math


# Importing module from the previous directory

loaded_model = tf.keras.models.load_model("image_based_sign_recognition_on_KU_dataset.h5")
tags = labels_list = [
    'Chandra Bindu',
    'Anusshar',
    'Bisharga',
    'Ka',
    'Kha',
    'Ga',
    'Gha',
    'Uo',
    'Ca',
    'Cha',
    'Borgio Ja/Anta Ja',
    'Jha',
    'Yo',
    'Ta',
    'Tha',
    'Da',
    'Dha',
    'Murdha Na/Donto Na',
    'ta/Khanda ta',
    'tha',
    'da',
    'dha',
    'pa',
    'fa',
    'Ba/Bha',
    'Ma',
    'Ba-y Ra/Da-y Ra/Dha-y Ra',
    'La',
    'Talobbo sha/Danta sa/Murdha Sha',
    'Ha'
]
framesPerSecond = 1
VIDEO_STREAM_LINK = "http:192.168.0.100:4747/video"
def collectHandKeypointsDataForClass():
    handDetector = HandDetector()
    IMG_SIZE = 300
    CROP_OFFSET = 30

    videoCaptureObject = cv2.VideoCapture(VIDEO_STREAM_LINK)

    frame_number = 0
    while True:
        success, frame = videoCaptureObject.read()
        if not success:
            print(f"Failed to read frame from video stream {VIDEO_STREAM_LINK}")
        frame_number+=1
        fps = videoCaptureObject.get(cv2.CAP_PROP_FPS)
        if frame_number % (math.ceil(fps/framesPerSecond))==0:
            temp = frame.copy()
            cv2.imshow(f"Capturing Video", temp)
            hands, annotatedFrame = handDetector.findHands(temp)
            if hands:
                oneHand = hands[0]
                x,y,w,h = oneHand['bbox']
                croppedImage = frame[y-CROP_OFFSET:y+h+CROP_OFFSET, x-CROP_OFFSET:x+w+CROP_OFFSET]
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
                image = cv2.resize(frame, (128, 128))
                image = np.array(image, dtype='float32') / 255.0
                image = image.reshape((1, 128, 128, 3))
                predictions = loaded_model.predict(image)
                print(predictions)
                predicted_class_index = np.argmax(predictions, axis=-1)
                cv2.putText(imageOfHand, f"{tags[predicted_class_index[0]]}, {predictions}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                cv2.imshow(f"Hand Detected", imageOfHand)
        if cv2.waitKey(1) == ord('q'):
            break
collectHandKeypointsDataForClass()
