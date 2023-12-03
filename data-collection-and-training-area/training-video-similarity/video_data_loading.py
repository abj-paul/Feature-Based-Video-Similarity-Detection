import os
import cv2
import numpy as np

def read_videos_and_extract_frames(data_directory):
    """
    Read videos from the specified directory, extract frames, and store them in a NumPy array.

    Parameters:
    - data_directory (str): The path to the directory containing video data.

    Returns:
    - frames_dict (dict): A dictionary where keys are category names and values are lists of frames.
    """
    frames_dict = {}

    # List all items in the data directory
    categories = os.listdir(data_directory)

    # Iterate over categories
    for category in categories:
        category_path = os.path.join(data_directory, category)

        # Check if the item is a directory
        if os.path.isdir(category_path):
            # List all items in the category directory (video files)
            videos = [video for video in os.listdir(category_path) if video.endswith('.mp4')]

            # Initialize an empty list to store frames for the current category
            category_frames = []

            # Iterate over videos in the current category
            for video_filename in videos:
                video_path = os.path.join(category_path, video_filename)

                # Open the video file
                cap = cv2.VideoCapture(video_path)

                # Read frames until there are no more frames
                success, frame = cap.read()
                while success:
                    # Convert the frame to grayscale if needed
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Append the frame to the list
                    category_frames.append(frame)

                    # Read the next frame
                    success, frame = cap.read()

                # Release the video capture object
                cap.release()

            # Convert the list of frames to a NumPy array and store in the dictionary
            frames_dict[category] = np.array(category_frames)

    return frames_dict

# Example usage:
data_directory = "../data"
frames_data = read_videos_and_extract_frames(data_directory)

