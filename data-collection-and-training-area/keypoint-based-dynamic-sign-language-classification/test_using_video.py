import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import sys 
sys.path.append("..")
from data_collection.collect_data_from_camera import get_keypoints
import mediapipe as mp

# Load the trained model
labels = ['Dhaka', 'Zuddho', 'Korat', 'Chittagong', 'Jela', 'Bivag', 'Gari', 'Taka', 'Phasi', 'Grephtar', 'Joma', 'Rong', 'Tahobil', 'Akashi', 'Kuthar', 'Guitar', 'Faridpur', 'Shotru', 'Sobuj', 'Jel']
model = load_model('video_classification_model.h5')

def preprocess_frame(frame, holistic_model, prev_keypoints: list):
    keypoint_frame, keypoints = get_keypoints(frame, holistic_model)
    prev_keypoints.append(keypoints)
    return keypoint_frame, prev_keypoints

def test_using_video(video_address):
    cap = cv2.VideoCapture(video_address) 
    keypoints_sequence = []

    mp_holistic = mp.solutions.holistic
    holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        keypoint_frame, keypoints_sequence = preprocess_frame(frame, holistic_model, keypoints_sequence)

        if np.array(keypoints_sequence).shape[0] == 16:
            predictions = model.predict(np.expand_dims(np.expand_dims(keypoints_sequence, axis=0), axis=-1))
            print(predictions)

            # Get the predicted class
            predicted_class = np.argmax(predictions)

            # Get the corresponding label
            predicted_label = labels[predicted_class]

            # Display the frame and predicted label
            print(f"{predicted_label}")
            keypoints_sequence = None
            return predicted_label

        cv2.imshow(f"{video_address}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

print(f"Class:[{labels}] ")
test_using_video("../data/Dhaka/Dhaka_0.mp4")
test_using_video("../data/Zuddho/Zuddho_0.mp4")
test_using_video("../data/Korat/Korat_0.mp4")




