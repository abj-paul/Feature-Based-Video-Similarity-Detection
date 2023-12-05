import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.preprocessing import StandardScaler


# Load your data
X = joblib.load("X.joblib")
Y = joblib.load("Y.joblib")
num_of_videos, num_frames_per_sample, num_keypoints = X.shape

# Reshape the data to 2D array for k-NN input
X_reshaped = X.reshape(-1, num_frames_per_sample * num_keypoints)

# Convert labels to categorical
label_encoder = LabelEncoder()
y_categorical = label_encoder.fit_transform(Y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)
print(f"X Shape: {X_train.shape}")

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the k-NN classifier on the standardized data
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)

# Evaluate the k-NN classifier on the standardized test set
accuracy = knn_classifier.score(X_test_scaled, y_test)
print(f"Accuracy after standardization: {accuracy}")

# Save the trained k-NN model
joblib.dump(knn_classifier, "knn_model.joblib")
