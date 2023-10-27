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

def collectActivityDataFromVideo(VIDEO_STREAM_LINK, className, numOfSamples, framesPerSample):
    videoCaptureObject = cv2.VideoCapture(VIDEO_STREAM_LINK)

    EVERY_N_FRAME = 3
    STARTING_FRAME = 1

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        if not os.path.exists("data"):
            os.mkdir("data")
        if not os.path.exists(f"data/{className}"):
            os.mkdir(f"data/{className}")
        for sample_number in range(numOfSamples):
            videoName = f"data/{className}_{sample_number}.mp4"
            out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc('m', 'p', '4', 'v') , 20.0, (640,480))

            frame_number = 0
            saved_frame_number = 0
            while True:
                success, frame = videoCaptureObject.read()
                if not success:
                    print(f"Error reading frames from videostream {VIDEO_STREAM_LINK}")
                    break

                if frame_number < STARTING_FRAME:
                    if not os.path.exists(f"data/{className}/sample{sample_number}"):
                        os.mkdir(f"data/{className}/sample{sample_number}")

                    saved_frame_number = 0

                elif saved_frame_number%EVERY_N_FRAME==0: 
                    keypointImage, keypoints = get_keypoints(frame, holistic)
                    np.save(f"data/{className}/sample{sample_number}/frame_{int(math.floor(saved_frame_number/EVERY_N_FRAME))}.npy", keypoints)
                
                if math.floor(saved_frame_number/EVERY_N_FRAME) > framesPerSample:
                    break
                frame_number+=1
                saved_frame_number+=1
                
                
                out.write(cv2.resize(frame, (640, 480)))
            out.release()
            return int(math.floor(saved_frame_number/EVERY_N_FRAME)) # BUG: WORKS ONLY FOR ONE VIDEO




# Video_Similarity.py

SAMPLE_NUM = 1
def load_activity_data(activities, FRAME_NUM):
    activity_dataset = []
    activity_labels = []
    for activity_index,activity in enumerate(activities):
        for sample_number in range(SAMPLE_NUM):
            single_sample = []
            for frame_number in range(1,FRAME_NUM-1):
                keypoints = np.load(f'data/{activity}/sample{sample_number}/frame_{frame_number}.npy')
                single_sample.append(keypoints)
            activity_dataset.append(single_sample)
            activity_labels.append(activity_index)
    return np.array(activity_dataset), to_categorical(activity_labels)


def cosine_similairty(a,b):
    return dot(a, b)/(norm(a)*norm(b))


def calculate_video_similarity(video1_address, video2_address):
    FRAME_NUM1 = collectActivityDataFromVideo(video1_address,"video1", 1, 300000)
    FRAME_NUM2 = collectActivityDataFromVideo(video2_address,"video2", 1, 300000)
    print(f"DEBUG: {FRAME_NUM1}, {FRAME_NUM2}")

    #FRAME_NUM1 = 635
    #FRAME_NUM2 = 635

    X,Y = load_activity_data(["video1", "video2"], min(FRAME_NUM1, FRAME_NUM2))
    print(f"DEBUG: {X.shape}, {X[0][0].shape}, {X[1][0].shape}")
    return cosine_similairty(X[0].flatten(), X[1].flatten())




