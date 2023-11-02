import cv2
import tensorflow as tf
from teachablemachinepose import TeachableMachinePose

# Load the Teachable Machine Pose model
model = TeachableMachinePose("./my_model/model.json")

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform pose estimation
    poses = model.predict(frame)

    # Draw the poses
    for pose in poses:
        # Draw keypoints and skeleton
        pose.draw(frame)

    # Display the frame with poses
    cv2.imshow("Teachable Machine Pose Model", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
        break

cap.release()
cv2.destroyAllWindows()

