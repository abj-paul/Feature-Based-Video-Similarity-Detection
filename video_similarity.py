import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from numpy import dot
from numpy.linalg import norm


SAMPLE_NUM = 30
FRAME_NUM = 30
def load_activity_data(activities):
    activity_dataset = []
    activity_labels = []
    for activity_index,activity in enumerate(activities):
        for sample_number in range(SAMPLE_NUM):
            single_sample = []
            for frame_number in range(1,FRAME_NUM):
                keypoints = np.load(f'data/{activity}/sample{sample_number}/frame_{frame_number}.npy')
                single_sample.append(keypoints)
            activity_dataset.append(single_sample)
            activity_labels.append(activity_index)
    return np.array(activity_dataset), to_categorical(activity_labels)


def cosine_similairt(a,b):
    return dot(a, b)/(norm(a)*norm(b))





