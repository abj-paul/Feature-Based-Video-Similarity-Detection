import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('video_classification_model.h5')

# Create a function to preprocess the frame before prediction
def preprocess_frame(frame):
    # Resize the frame to match the input size of the model without adding a new dimension
    resized_frame = cv2.resize(frame, (1662, 16))

    # Convert to grayscale if needed
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values to be between 0 and 1
    #normalized_frame = gray_frame / 255.0

    # Add an additional dimension for the channel (assuming 1 channel for grayscale)
    preprocessed_frame = np.expand_dims(normalized_frame, axis=-1)

    return preprocessed_frame


# Open the camera
cap = cv2.VideoCapture("http://192.168.0.101:4747/video")  # Use 0 for the default camera

while True:
    # Capture a frame
    ret, frame = cap.read()
    
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    
    # Make a prediction
    predictions = model.predict(np.expand_dims(preprocessed_frame, axis=0))
    
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