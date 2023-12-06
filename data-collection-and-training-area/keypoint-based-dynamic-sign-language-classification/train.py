import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import joblib

# Load your data
X = joblib.load("X.joblib")
Y = joblib.load("Y.joblib")
num_of_videos, num_frames_per_sample, num_keypoints = X.shape
NUM_CLASSES = 10

# Reshape the data to 3D array for LSTM input
X_reshaped = X.reshape(-1, num_frames_per_sample, num_keypoints)

# Convert labels to categorical
label_encoder = LabelEncoder()
y_categorical = to_categorical(label_encoder.fit_transform(Y))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)
print(f"X Shape: {X_train.shape}")

# Create a model with LSTM and Conv1D layers
model = Sequential()
model.add(LSTM(64, input_shape=(num_frames_per_sample, num_keypoints), return_sequences=True))
model.add(Conv1D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
accuracy = model.evaluate(X_test, y_test)[1]
print(f"Accuracy: {accuracy}")

# Save the model
model.save("video_classification_model.h5")
