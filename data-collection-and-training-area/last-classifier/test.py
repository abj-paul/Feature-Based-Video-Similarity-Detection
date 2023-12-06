import cv2
import numpy as np
from tensorflow.keras.models import load_model

def read_single_video(video_path, MAX_FRAME_PER_VIDEO=16):
    cap = cv2.VideoCapture(video_path)

    success, frame = cap.read()
    video_frames = []
    while success:
        if len(video_frames) == MAX_FRAME_PER_VIDEO:
            break
        resized_frame = cv2.resize(frame, (128, 128))
        resized_frame = resized_frame / 255.0
        video_frames.append(resized_frame)

        success, frame = cap.read()
    cap.release()

    return np.array(video_frames)

# Read a single video and expand the dimensions to match the model's input shape
video_data = read_single_video('../data/Akashi/Akashi_0.mp4')
video_data = np.expand_dims(video_data, axis=0)

print(f"Input shape for prediction: {video_data.shape}")

# Load the pre-trained model
model = load_model('video_classification_model.h5')

# Make predictions
predictions = model.predict(video_data)
print(f"Predictions: {predictions}")
print(f"Predictions: {np.argmax(predictions)}")

