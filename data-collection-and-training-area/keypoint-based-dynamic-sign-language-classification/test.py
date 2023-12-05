import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import sys 
sys.path.append("..")
from data_collection.collect_data_from_camera import get_keypoints
import mediapipe as mp

# Load the trained model
labels = joblib.load("labels.joblib")
model = load_model('video_classification_model.h5')

# Function to get keypoints and preprocess the frame
def preprocess_frame(frame, holistic_model):
    resized_frame = cv2.resize(frame, (1662, 16))
    keypoint_frame, keypoints = get_keypoints(resized_frame, holistic_model)
    return keypoint_frame, keypoints

# Open the camera
cap = cv2.VideoCapture("http://192.168.0.101:4747/video")  # Use 0 for the default camera

# Set the frame skipping interval
frame_skip_interval = 5  # Skip every 5 frames

frame_count = 0

# Holistic model for keypoints
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    # Capture a frame
    ret, frame = cap.read()

    # Increment the frame count
    frame_count += 1

    # Skip frames if needed
    if frame_count % frame_skip_interval != 0:
        continue

    # Preprocess the frame and get keypoints
    keypoint_frame, keypoints = preprocess_frame(frame, holistic_model)

    # Make a prediction
    predictions = model.predict(keypoints)
    print(predictions)

    # Get the predicted class
    predicted_class = np.argmax(predictions)

    # Get the corresponding label
    predicted_label = labels[predicted_class]

    # Display the frame and predicted label
    cv2.putText(keypoint_frame, f"Predicted Label: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video Classification', keypoint_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
