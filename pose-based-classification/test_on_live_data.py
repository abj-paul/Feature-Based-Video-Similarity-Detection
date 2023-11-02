import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os
import sys

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

frame_number = 0
predicted_class_index = 0
predictions = ["Initializing.."]
EVERY_N_FRAME = 5
VIDEO_STREAM_LINK = "http://192.168.0.100:4747/video"

videoCaptureObject = cv2.VideoCapture(VIDEO_STREAM_LINK)
while True:
    success, frame = videoCaptureObject.read()
    if not success:
        print(f"Failed to read frame from video stream {VIDEO_STREAM_LINK}")
        break
    frame_number += 1

    if frame_number % EVERY_N_FRAME == 0:
        image = cv2.resize(frame, (128, 128))
        image = np.array(image, dtype='float32') / 255.0
        image = image.reshape((1, 128, 128, 3))  # Reshape to match the model's input shape
        print(image.shape)

        predictions = loaded_model.predict(image)
        predicted_class_index = np.argmax(predictions, axis=-1)
        cv2.putText(frame, f"{tags[predicted_class_index[0]]}, {predictions}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.imshow("Predictions", frame)
    cv2.imshow("Live", frame)

    # Ensure the OpenCV window is properly sized to match the frame
    cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live", frame.shape[1], frame.shape[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
videoCaptureObject.release()
cv2.destroyAllWindows()

