import cv2 
import mediapipe as mp 
import numpy as np
import os
import math 

VIDEO_STREAM_LINK = "http://192.168.0.104:4747/video"

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

def collectActivityDataFromCamera(className, numOfSamples, framesPerSample):
    videoCaptureObject = cv2.VideoCapture(VIDEO_STREAM_LINK)

    EVERY_N_FRAME = 3
    STARTING_FRAME = 100

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        cv2.waitKey(3000)
        if not os.path.exists("data"):
            os.mkdir("data")
        if not os.path.exists(f"data/{className}"):
            os.mkdir(f"data/{className}")
        for sample_number in range(numOfSamples):
            noisyVideoName = f"data/noisy_{className}_{sample_number}.mp4"
            videoName = f"data/{className}_{sample_number}.mp4"
            out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc('m', 'p', '4', 'v') , 20.0, (640,480))
            outNoisy = cv2.VideoWriter(noisyVideoName, cv2.VideoWriter_fourcc('m', 'p', '4', 'v') , 20.0, (640,480))

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

                    cv2.putText(frame, f"STARTING ACTIVITY SAMPLE {sample_number}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
                    cv2.imshow("Live Video", frame)
                    saved_frame_number = 0

                elif saved_frame_number%EVERY_N_FRAME==0: 
                    keypointImage, keypoints = get_keypoints(frame, holistic)
                    cv2.putText(keypointImage, f"COLLECTING ACTIVITY SAMPLE {sample_number} FOR {className}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
                    np.save(f"data/{className}/sample{sample_number}/frame_{int(math.floor(saved_frame_number/EVERY_N_FRAME))}.npy", keypoints)

                    cv2.imshow("Live Video", keypointImage)
                    out.write(cv2.resize(frame, (640, 480)))
                    outNoisy.write(cv2.resize(frame, (640, 480)))
                else: 
                    outNoisy.write(cv2.resize(frame, (640, 480)))
                
                if cv2.waitKey(1) == ord('q'): 
                    videoCaptureObject.release()
                    cv2.destroyAllWindows()
                    break
                if math.floor(saved_frame_number/EVERY_N_FRAME) > framesPerSample:
                    break
                frame_number+=1
                saved_frame_number+=1
                
                

            out.release()
            outNoisy.release()


