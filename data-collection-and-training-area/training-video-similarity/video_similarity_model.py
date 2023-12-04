from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import tensorflow as tf
import joblib

# Load Data
X = joblib.load("X.joblib")
Y = joblib.load("Y.joblib")
print(f"X={X.shape} Y={Y.shape}")

# Define the model
model = Sequential()

# Input layer
model.add(Dense(512, input_shape=(X.shape[1],), activation='relu'))

# Hidden layers
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Dropout layer to prevent overfitting

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(1, activation='sigmoid'))  # Binary classification, so using sigmoid activation

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Model training
history = model.fit(
    X,  # input data
    Y,  # labels
    epochs=1,  # adjust as needed
    batch_size=10,  # adjust as needed (should be the number of videos)
    validation_split=0.2,  # adjust as needed, this is for validation split
    verbose=1
)

# Save the model
model.save('video_similarity_model.h5')

