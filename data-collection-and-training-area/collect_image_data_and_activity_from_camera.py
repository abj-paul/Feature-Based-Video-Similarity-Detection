import cv2 
import mediapipe as mp 
import numpy as np
import os
import math 

VIDEO_STREAM_LINK = "http://192.168.0.100:4747/video"

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

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


def collectDataForClass(className, number_of_sample):
    framesPerSecond=0.5
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists(f'data/{className}'):
        os.mkdir(f'data/{className}')
    
    videoCaptureObject = cv2.VideoCapture(VIDEO_STREAM_LINK)
    frame_number = 0
    out = cv2.VideoWriter(f"data/{className}.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v') , 20.0, (640,480))
    out_keypoint_image = cv2.VideoWriter(f"data/{className}_keypoint.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v') , 20.0, (640,480))

    starting_frame = 200
    saved_frame_number = 0
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, frame = videoCaptureObject.read()
            if not success: 
                print(f"Failed to read frame from video stream {VIDEO_STREAM_LINK}")
                break
            frame_number+=1
            fps = videoCaptureObject.get(cv2.CAP_PROP_FPS)
            
            if frame_number < starting_frame: 
                    cv2.imshow("Sample", frame)
            elif frame_number % (math.ceil(fps/framesPerSecond))==0:
                    keypointImage, keypoints = get_keypoints(frame, holistic)
                    np.save(f"data/{className}/frame_{saved_frame_number}.npy", keypoints)
                    out.write(cv2.resize(frame, (640, 480)))
                    out_keypoint_image.write(cv2.resize(keypointImage, (640, 480)))
                    saved_frame_number+=1
                    cv2.putText(keypointImage, f"COLLECTING IMAGE SAMPLE {saved_frame_number} FOR {className}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
                    cv2.imshow("Sample", keypointImage)
            cv2.imshow("Capturing Image", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            if saved_frame_number > number_of_sample: 
                break

collectDataForClass("miser", 30)
