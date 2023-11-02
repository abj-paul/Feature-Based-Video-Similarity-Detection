import tensorflow as tf
from tensorflow import keras
import cv2 
import numpy as np
import os
import sys

# Importing module from previous directory
script_directory = os.path.dirname(os.path.abspath(__file__))
module_directory = os.path.join(script_directory, '..')
sys.path.append(module_directory)
from collect_image_data_and_activity_from_camera import get_keypoints
import mediapipe as mp 
mp_holistic = mp.solutions.holistic

loaded_model = tf.keras.models.load_model("keypoints_based_sign_recognition_model.h5")
tags = ["Camera", "miser", "Surprise"]


frame_number = 0
predicted_class_index = 0
predictions = ["Initializing.."]
EVERY_N_FRAME = 5
VIDEO_STREAM_LINK = "http://192.168.0.100:4747/video"

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    videoCaptureObject = cv2.VideoCapture(VIDEO_STREAM_LINK)
    while True:
        success, frame = videoCaptureObject.read()
        '''
        if not success: 
            print(f"Failed to read frame from video stream {VIDEO_STREAM_LINK}")
            break
        frame_number+=1

        if frame_number % EVERY_N_FRAME == 0:
            keypointImage, keypoints = get_keypoints(frame, holistic)

            predictions = loaded_model.predict(keypoints.reshape(1, -1))
            print(predictions)
            predicted_class_index = np.argmax(predictions, axis=-1)
            cv2.putText(keypointImage, f"{predicted_class_index}, {predictions}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
            cv2.imshow("Predictions", keypointImage)
            '''
        cv2.imshow("Live", frame)


