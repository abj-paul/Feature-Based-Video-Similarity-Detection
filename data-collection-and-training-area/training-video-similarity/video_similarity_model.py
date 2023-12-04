from video_data_loading import read_videos_and_extract_frames, construct_dataset_for_video_similarity

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# Loading Data
data_directory = "../data"
videos, labels = read_videos_and_extract_frames(data_directory)
X,Y = construct_dataset_for_video_similarity(videos, labels)
print(f"X={X.shape} Y={Y.shape}")

# Define the model
model = Sequential()

# Add a dense layer with a sigmoid activation function for binary classification
model.add(Dense(units=1, activation='sigmoid', input_shape=(1638400,)))
with tf.device('/cpu:0'):
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X, Y, epochs=10, batch_size=1, validation_split=0.2)
