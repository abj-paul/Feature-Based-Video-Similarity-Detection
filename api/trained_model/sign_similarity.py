import cv2 
import mediapipe as mp 
import numpy as np
from numpy import dot
import os
import math 

from tensorflow.keras.utils import to_categorical
from numpy.linalg import norm



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def drawKeyPointsOnImage(image, results):
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.
        get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.
        get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.
        get_default_pose_landmarks_style())
    return image


def get_keypoints(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image) # Time consuming.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return drawKeyPointsOnImage(image, results), build1DArrayFromResults(results)

def build1DArrayFromResults(results):
    pose = np.array([[entry.x, entry.y, entry.z, entry.visibility] for entry in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[entry.x, entry.y, entry.z] for entry in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    leftHand = np.array([[entry.x, entry.y, entry.z] for entry in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rightHand = np.array([[entry.x, entry.y, entry.z] for entry in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose,face,leftHand, rightHand])


def cosine_similairty(a,b):
    return dot(a, b)/(norm(a)*norm(b))

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
def calculate_image_sign_similarity(image1_address, image2_address):
    image1 = cv2.imread(image1_address)
    image2 = cv2.imread(image2_address)

    keypoint_marked_image1, keypointArr1 = get_keypoints(image1, holistic)
    keypoint_marked_image2, keypointArr2 = get_keypoints(image2, holistic)

    cv2.imwrite("data/a.png", keypoint_marked_image1)
    cv2.imwrite("data/b.png", keypoint_marked_image2)

    return cosine_similairty(keypointArr1, keypointArr2), "data/a.png", "data/b.png" 

#print(calculate_image_sign_similarity("../Jha.png", "../Jha.png")[0])





