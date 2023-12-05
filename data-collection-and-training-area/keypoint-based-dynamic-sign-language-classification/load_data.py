import numpy as np
import os
import joblib

def __load_single_sample(sample_path, NUM_OF_FRAME_PER_SAMEPLE=17):
    keypoints_collection = []
    for num in range(1, NUM_OF_FRAME_PER_SAMEPLE):
        file_path = f"{sample_path}/frame_{num}.npy"
        keypoints_collection.append(np.load(file_path))
    return np.array(keypoints_collection)

def __load_all_samples_for_a_class(class_path, NUM_OF_SAMPLES=20):
    video_collection = []
    for num in range(1,NUM_OF_SAMPLES):
        video_collection.append(__load_single_sample(f"{class_path}/sample{num}"))
    return np.array(video_collection)

def __list_directories(folder_path):
    entries = os.listdir(folder_path)
    directories = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]
    return directories

def load_all_keypoint_video_from_BdSL_420_dataset(dataset_path):
    X = None
    class_list = __list_directories(dataset_path)
    Y = []
    for index, class_name in enumerate(class_list):
        new_sample = __load_all_samples_for_a_class(f"{dataset_path}/{class_name}")
        Y = Y + [class_name for _ in new_sample]
        if index==0:
            X = new_sample
        else:
            X = np.concatenate((X, new_sample), axis=0)
    return np.array(X), Y

X,Y = load_all_keypoint_video_from_BdSL_420_dataset("../data")
joblib.dump(X, "X.joblib")
joblib.dump(Y, "Y.joblib")
print(f"X={X.shape}, Y={len(Y)}")
#print(__load_single_sample("../data/Akashi/sample0").shape) # 20 X 1662
#print(__load_all_samples_for_a_class("../data/Akashi/").shape) # 19 X 20 X 1662
#print(load_all_keypoint_video_from_BdSL_420_dataset("../data")[0].shape) # (21, 19, 16, 1662)

