import cv2
import mediapipe as mp
import numpy as np
import os
import math
import os

VIDEO_STREAM_LINK = "http://192.168.0.100:4747/video"

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def get_image_paths_from_KU_dataset(root_directory):
    directory_image_paths = []

    for dirpath, dirnames, filenames in os.walk(root_directory):
        image_paths = [os.path.join(dirpath, filename) for filename in filenames if filename.endswith('.jpg')]
        if image_paths:
            directory_name = os.path.basename(dirpath)
            directory_image_paths.append((directory_name, image_paths))
    return directory_image_paths


def _drawKeyPointsOnImage(image, keypoints):
    mp_drawing.draw_landmarks(
        image,
        keypoints.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
        image,
        keypoints.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.
        get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        keypoints.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.
        get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        keypoints.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.
        get_default_pose_landmarks_style())
    return image



def _build1DArrayFromResults(keypoints):
    pose = np.array([[entry.x, entry.y, entry.z, entry.visibility] for entry in keypoints.pose_landmarks.landmark]).flatten() if keypoints.pose_landmarks else np.zeros(33*4)
    face = np.array([[entry.x, entry.y, entry.z] for entry in keypoints.face_landmarks.landmark]).flatten() if keypoints.face_landmarks else np.zeros(468*3)
    leftHand = np.array([[entry.x, entry.y, entry.z] for entry in keypoints.left_hand_landmarks.landmark]).flatten() if keypoints.left_hand_landmarks else np.zeros(21*3)
    rightHand = np.array([[entry.x, entry.y, entry.z] for entry in keypoints.right_hand_landmarks.landmark]).flatten() if keypoints.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose,face,leftHand, rightHand])

def get_keypoints(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    keypoints = model.process(image) # Time consuming.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return _drawKeyPointsOnImage(image, keypoints), _build1DArrayFromResults(keypoints)


def save_keypoints_and_image(image_path, class_name, sample_number):
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists(f"data/{class_name}"):
        os.mkdir(f"data/{class_name}")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image = cv2.imread(image_path)
        keypointImage, keypoints = get_keypoints(image, holistic)
        np.save(f"data/{class_name}/sample_{sample_number}.npy", keypoints)
        cv2.imwrite(f"data/{class_name}/keypoint_sample_{sample_number}.jpg",cv2.resize(image, (640, 480)))


directory_image_paths = get_image_paths_from_KU_dataset("MSLD/")
for directory_name, folder_images in directory_image_paths:
    for index,image_path in enumerate(folder_images):
        save_keypoints_and_image(image_path, directory_name, index)
        print(f"Processing {image_path}")

