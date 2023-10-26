import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from numpy import dot
from numpy.linalg import norm

from collect_data_from_video import collectActivityDataFromVideo


SAMPLE_NUM = 1
def load_activity_data(activities, FRAME_NUM):
    activity_dataset = []
    activity_labels = []
    for activity_index,activity in enumerate(activities):
        for sample_number in range(SAMPLE_NUM):
            single_sample = []
            for frame_number in range(1,FRAME_NUM-1):
                keypoints = np.load(f'data/{activity}/sample{sample_number}/frame_{frame_number}.npy')
                single_sample.append(keypoints)
            activity_dataset.append(single_sample)
            activity_labels.append(activity_index)
    return np.array(activity_dataset), to_categorical(activity_labels)


def cosine_similairty(a,b):
    return dot(a, b)/(norm(a)*norm(b))


def calculate_video_similarity(video1_address, video2_address):
    #FRAME_NUM1 = collectActivityDataFromVideo(video1_address,"video1", 1, 300000)
    #FRAME_NUM2 = collectActivityDataFromVideo(video2_address,"video2", 1, 300000)
    #print(f"DEBUG: {FRAME_NUM1}, {FRAME_NUM2}")

    FRAME_NUM1 = 635
    FRAME_NUM2 = 635

    X,Y = load_activity_data(["video1", "video2"], min(FRAME_NUM1, FRAME_NUM2))
    print(f"DEBUG: {X.shape}, {X[0][0].shape}, {X[1][0].shape}")
    return cosine_similairty(X[0].flatten(), X[1].flatten())





