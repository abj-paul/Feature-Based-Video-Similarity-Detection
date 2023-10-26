from collect_data_from_camera import *


print("Testing Keypoint Generation For Image....")
cv2.imwrite("test_keypoint.jpg",get_keypoints(cv2.imread("PXL_20230920_051842270.jpg"), mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5))[0])
print("Testing - Collecting Data From Camera....")
collectActivityDataFromVideo("Shuvo Sokal", 5, 30)