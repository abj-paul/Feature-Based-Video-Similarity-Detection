import numpy as np
from tensorflow.keras.utils import to_categorical


def load_class_data(activities, SAMPLE_NUM):
    class_dataset = []
    class_labels = []
    for class_index, class_name in enumerate(activities):
        for sample_number in range(1,SAMPLE_NUM):
            keypoints = np.load(f'data/{class_name}/frame_{sample_number}.npy')
            class_dataset.append(keypoints)
            class_labels.append(class_index)
    return np.array(class_dataset), to_categorical(class_labels)

X,Y = load_class_data(["Camera","miser","Surprise"],100)
print(X.shape)
print(Y.shape)

