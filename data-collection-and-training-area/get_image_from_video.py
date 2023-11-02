import cv2
import os

# Input video file path
video_path = 'data/Surprise.mp4'

# Output directory to save frames
output_dir = 'output_frames/Surprise'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if we've reached the end of the video

    # Save the frame as an image in the output directory
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

# Release the video capture object
cap.release()

print(f"Frames extracted: {frame_count}")
print(f"Frames saved in: {output_dir}")

