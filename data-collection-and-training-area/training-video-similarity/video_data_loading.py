import os
import cv2
import numpy as np
import random
import joblib

def compare_videos(class_a, class_b):
    return np.array_equal(class_a, class_b)

# Shape: 50 X 480 X 640 X 3 = 50 X 921600
import numpy as np
def construct_dataset_for_video_similarity(videos, labels, max_data_per_class=3):
    video_pairs = []
    pSame = []
    OFFSET = 20
    for index in range(len(labels)-OFFSET):
            video_pairs.append(np.concatenate((videos[index].reshape(-1), videos[index+OFFSET].reshape(-1))))
            pSame.append(labels[index]==labels[index+OFFSET])
    OFFSET = 1
    for index in range(len(labels)-OFFSET):
            video_pairs.append(np.concatenate((videos[index].reshape(-1), videos[index+OFFSET].reshape(-1))))
            pSame.append(labels[index]==labels[index+OFFSET])

    true_count = sum([1 for value in pSame if value])
    false_count = sum([1 for value in pSame if not value])
    print(f"Data balancing----\nN(True)={true_count}\nN(False)={false_count}")

    return np.array(video_pairs), np.array(pSame).reshape(-1, 1)


    

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


def read_videos_and_extract_frames(data_directory, num_frames_per_video=15, num_videos_per_class=10):
    print("Reading videos")
    videos_loaded = []
    labels_for_each_video = []

    # List all items in the data directory
    categories = os.listdir(data_directory)

    # Iterate over categories
    for category in categories:
        category_path = os.path.join(data_directory, category)

        # Check if the item is a directory
        if os.path.isdir(category_path):
            # List all items in the category directory (video files)
            videos = [video for video in os.listdir(category_path) if video.endswith('.mp4')]

            # Randomly choose at most 5 videos per category
            selected_videos = random.sample(videos, min(num_videos_per_class, len(videos)))

            # Initialize an empty list to store frames for the current category

            # Iterate over selected videos in the current category
            for video_filename in selected_videos:
                video_path = os.path.join(category_path, video_filename)

                cap = cv2.VideoCapture(video_path)

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Randomly choose num_frames_per_video frames
                #selected_frames = sorted(random.sample(range(total_frames), num_frames_per_video))
                selected_frames = [num for num in range(num_frames_per_video)]
                video_frames = []
                for frame_num in selected_frames:
                    # Set the frame position
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

                    # Read the selected frame
                    success, frame = cap.read()

                    if success:
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        # Resize the grayscale image
                        resized_frame = cv2.resize(gray_frame, (128, 128))

                        video_frames.append(resized_frame)
                videos_loaded.append(np.array(video_frames))
                labels_for_each_video.append(category)

                cap.release()
            
                #print(f"Video Size: {len(category_frames)}, Frame Size: {frame.shape}")
    return videos_loaded, labels_for_each_video

            

# Example usage:
data_directory = "../data"
videos, labels = read_videos_and_extract_frames(data_directory)
X,Y = construct_dataset_for_video_similarity(videos, labels)
print(f"X={X.shape} Y={Y.shape}")
joblib.dump(X,"X.joblib")
joblib.dump(Y,"Y.joblib")