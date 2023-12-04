import numpy as np
from keras.models import load_model
import joblib
import cv2
import random
import os
# Load the trained model
model = load_model('video_similarity_model.h5')

def test_for_all_loaded_vides():
    # Load test data
    X_test = joblib.load("X.joblib")  
    Y_test = joblib.load("Y.joblib")  

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Convert predictions to binary values (0 or 1) based on a threshold (e.g., 0.5)
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)

    # Evaluate the model on the test data
    accuracy = np.mean(binary_predictions == Y_test)
    print(f"Accuracy on test data: {accuracy}")

def preprocess_video(video_name, num_frames_per_video):
    category_frames = []

    cap = cv2.VideoCapture(video_name)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Check if the requested number of frames is valid
    if num_frames_per_video > total_frames:
        raise ValueError("num_frames_per_video is larger than the total number of frames in the video")

    # Randomly choose num_frames_per_video frames
    selected_frames = sorted(random.sample(range(total_frames), num_frames_per_video))

    for frame_num in selected_frames:
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        # Read the selected frame
        success, frame = cap.read()

        if success:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize the grayscale image
            resized_frame = cv2.resize(gray_frame, (128, 128))
            flattened_frame = resized_frame.reshape(-1)  # Flatten the image

            category_frames.append(flattened_frame)

    # Release the video capture object
    cap.release()

    return np.concatenate(category_frames), video_name

def test_for_two_video(video1_name, video2_name, num_frames_per_video=15):
    video1, label1 = preprocess_video(video1_name, num_frames_per_video)
    video2, label2 = preprocess_video(video2_name, num_frames_per_video)

    video_pair = np.concatenate((video1, video2))
    pSame = label1 == label2

    # Assuming 'model' is already loaded and defined
    predictions = model.predict(video_pair.reshape(1, -1))  # Reshape to add batch dimension

    # Convert predictions to binary values (0 or 1) based on a threshold (e.g., 0.5)
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)

    # Determine if the model's prediction is correct
    prediction_correct = binary_predictions[0] == pSame

    # Print the results
    print(f"Video Pair: {video1_name} and {video2_name}")
    print(f"True Labels: Same={pSame}, Different={not pSame}")
    print(f"Model Predictions: Same={binary_predictions[0]}, Different={not binary_predictions[0]}")
    print(f"Prediction Correct: {prediction_correct}\n")

def test_for_all_videos(data_directory):
    categories = os.listdir(data_directory)

    # Iterate over categories
    all_videos = []
    for category in categories:
        category_path = os.path.join(data_directory, category)

        # Check if the item is a directory
        if os.path.isdir(category_path):
            # List all items in the category directory (video files)
            for video in os.listdir(category_path):
                if video.endswith('.mp4'):
                    video_path = os.path.join(category_path, video)
                    all_videos.append(video_path)
    
    max_test_count = 100
    for index, video_file_path in enumerate(all_videos):
        if index>=max_test_count: break
        test_for_two_video(all_videos[index], all_videos[index+20])


# # Example usage
# test_for_two_video("../data/Akashi/Akashi_2.mp4", "../data/Akashi/Akashi_5.mp4")
# test_for_two_video("../data/Akashi/Akashi_2.mp4", "../data/Bivag/Bivag_3.mp4")

test_for_all_videos("../data/")

