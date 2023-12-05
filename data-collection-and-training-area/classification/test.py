import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('video_classification_model.h5')

# Create a function to preprocess the frame before prediction
def preprocess_frame(frame):
    # Resize the frame to match the input size of the model without adding a new dimension
    resized_frame = cv2.resize(frame, (16384, 1))
    
    # Convert to grayscale if needed
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    
    # Normalize pixel values to be between 0 and 1
    #preprocessed_frame = gray_frame / 255.0
    
    # Add an additional dimension for the channel (assuming 1 channel for grayscale)
    
    return gray_frame

# Open the camera
cap = cv2.VideoCapture("http://192.168.0.101:4747/video")  # Use 0 for the default camera

# Set the frame skip interval
frame_skip_interval = 5  # Adjust as needed

while True:
    # Capture a frame
    ret, frame = cap.read()
    
    # Skip frames at regular intervals
    for _ in range(frame_skip_interval):
        ret, _ = cap.read()
    
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    
    # Make a prediction
    predictions = model.predict(preprocessed_frame)
    
    # Get the predicted class
    predicted_class = np.argmax(predictions)
    
    # Display the frame and predicted class
    cv2.putText(frame, f"Predicted Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video Classification', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
