import os 
import numpy as np 
import cv2

def read_videos_and_extract_frames(data_directory, MAX_FRAME_PER_VIDEO=16):
    print(f"Loading video dataset...")
    X = []
    Y = []

    # List all items in the data directory
    categories = os.listdir(data_directory)

    # Iterate over categories
    for category in categories:
        category_path = os.path.join(data_directory, category)

        # Check if the item is a directory
        if os.path.isdir(category_path):
            # List all items in the category directory (video files)
            videos = [video for video in os.listdir(category_path) if video.endswith('.mp4')]

            # Iterate over videos in the current categor
            for video_filename in videos:
                video_path = os.path.join(category_path, video_filename)

                cap = cv2.VideoCapture(video_path)

                success, frame = cap.read()
                video_frames = []
                while success:
                    if len(video_frames)==MAX_FRAME_PER_VIDEO: break
                    #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Resize the grayscale image
                    resized_frame = cv2.resize(frame, (128, 128))
                    resized_frame = resized_frame / 255.0
                    video_frames.append(resized_frame)

                    success, frame = cap.read()
                cap.release()

                X.append(np.array(np.array(video_frames)))
                Y.append(category)

    return np.array(X),np.array(Y)

X,Y = read_videos_and_extract_frames("../data")
print(f"X={X.shape}, Y={Y.shape}")

import joblib 
joblib.dump(X,"X.joblib")
joblib.dump(Y,"Y.joblib")