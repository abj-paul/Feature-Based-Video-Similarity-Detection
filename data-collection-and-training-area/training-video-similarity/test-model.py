import numpy as np
from keras.models import load_model
import joblib

# Load the trained model
model = load_model('video_similarity_model.h5')

# Load test data
X_test = joblib.load("X.joblib")  # Replace 'X_test.joblib' with the filename for your test data
Y_test = joblib.load("Y.joblib")  # Replace 'Y_test.joblib' with the filename for your test labels

# Make predictions on the test data
predictions = model.predict(X_test)

# Convert predictions to binary values (0 or 1) based on a threshold (e.g., 0.5)
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)

# Evaluate the model on the test data
accuracy = np.mean(binary_predictions == Y_test)
print(f"Accuracy on test data: {accuracy}")

# You can also use model.evaluate for more metrics
# evaluation = model.evaluate(X_test, Y_test)
# print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")
