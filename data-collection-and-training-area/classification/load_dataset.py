import os
import cv2
import numpy as np
import joblib 

def compare_videos(class_a, class_b):
    return np.array_equal(class_a, class_b)

def load_videos_to_training(video_dictionary, MAX_VIDEOS_PER_CLASS = 10):
    print("Constructing classification dataset...")
    X = []
    Y = []
    for classname, videos in video_dictionary.items():
        for index in range(MAX_VIDEOS_PER_CLASS):
            X.append(videos[index].reshape(-1))
            Y.append(classname)
    
    unique_class_count = len(set(Y))
    print(f"Unique Class  Count: %d" % unique_class_count)

    return np.array(X), np.array(Y).reshape(-1,1)





def iterate_videos_and_compare(video_dictionary):
    video_keys = list(video_dictionary.keys())

    for index1, key1 in enumerate(video_keys):
        for index2, key2 in enumerate(video_keys):
            if index1 == index2:
                video1 = video_dictionary[key1]
                video2 = video_dictionary[key2]
                print(key1, key2)
                result = compare_videos(video1, video2)
                print("Comparison betweenc" + key1 + " and " + key2 + " " + str(result))
        


def read_videos_and_extract_frames(data_directory):
    print(f"Loading video dataset...")
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

                cap = cv2.VideoCapture(video_path)

                success, frame = cap.read()
                while success:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Resize the grayscale image
                    resized_frame = cv2.resize(gray_frame, (128, 128))
                    category_frames.append(resized_frame)

                    success, frame = cap.read()

                cap.release()

            frames_dict[category] = np.array(category_frames)

    return frames_dict

# Example usage:
data_directory = "../data"
frames_data = read_videos_and_extract_frames(data_directory)
X,Y = load_videos_to_training(frames_data)
print(f"X={X.shape}, Y={Y.shape}")


joblib.dump(X, "X.joblib")
joblib.dump(Y, "Y.joblib")