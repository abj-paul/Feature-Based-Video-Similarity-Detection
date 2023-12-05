import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import sys 
sys.path.append("..")
from data_collection.collect_data_from_camera import get_keypoints
import mediapipe as mp

# Load the trained model
labels = joblib.load("Y.joblib")
model = load_model('video_classification_model.h5')

# Function to get keypoints and preprocess the frame
def preprocess_frame(frame, holistic_model, prev_keypoints):
    keypoint_frame, keypoints = get_keypoints(frame, holistic_model)
    prev_keypoints.append(keypoints)
    return keypoint_frame, prev_keypoints

# Open the camera
cap = cv2.VideoCapture("http://192.168.0.101:4747/video")  # Use 0 for the default camera

# Set the frame skipping interval
frame_skip_interval = 5  # Skip every 5 frames

frame_count = 0
keypoints_sequence = []

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

    # Preprocess the frame and accumulate keypoints
    keypoint_frame, keypoints_sequence = preprocess_frame(frame, holistic_model, keypoints_sequence)

    # Make a prediction when enough frames are accumulated
    if len(keypoints_sequence) == 16:
        keypoint_frame = np.array(keypoint_frame)
        predictions = model.predict(np.expand_dims(np.expand_dims(keypoints_sequence, axis=0), axis=-1))
        print(predictions)

        # Get the predicted class
        predicted_class = np.argmax(predictions)

        # Get the corresponding label
        predicted_label = labels[predicted_class]

        # Display the frame and predicted label
        cv2.putText(keypoint_frame, f"Predicted Label: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"{predicted_label}")
        keypoints_sequence = []
    else:
        cv2.putText(keypoint_frame, f"Accumulated Frames: {len(keypoints_sequence)}/16", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Video Classification', keypoint_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
