import cv2
import mediapipe as mp
import os
import numpy as np
import sys 
sys.path.append("..")
from preprocess import *
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
# For static images:
# IMAGE_FILES = os.listdir("Image")
# BG_COLOR = (192, 192, 192) # gray
# root="Image"
# with mp_pose.Pose(
#     static_image_mode=True,
#     model_complexity=2,
#     enable_segmentation=True,
#     min_detection_confidence=0.5) as pose:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(os.path.join(root,file))
#     image_height, image_width, _ = image.shape
#     # Convert the BGR image to RGB before processing.
#     results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     if not results.pose_landmarks:
#       continue
#     print(
#         f'Nose coordinates: ('
#         f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
#         f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
#     )

#     annotated_image = image.copy()
#     # Draw segmentation on the image.
#     # To improve segmentation around boundaries, consider applying a joint
#     # bilateral filter to "results.segmentation_mask" with "image".
#     condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
#     bg_image = np.zeros(image.shape, dtype=np.uint8)
#     bg_image[:] = BG_COLOR
#     annotated_image = np.where(condition, annotated_image, bg_image)
#     # Draw pose landmarks on the image.
#     mp_drawing.draw_landmarks(
#         annotated_image,
#         results.pose_landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#     cv2.imwrite('./annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    # mp_drawing.plot_landmarks(
    #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# # For webcam input:
rtsp_link="rtsp://admin:XEJVQU@192.168.1.3:554"
video_link="/home/cuong/Documents/Project/Round1_mediapipe/GH010354_5_378_3750.avi"
cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():

      # Read feed
      ret, frame = cap.read()
      frame=cv2.resize(frame,(1280,720))

      # Make detections
      image, results = mediapipe_detection(frame, holistic)
      
      
      # Draw landmarks
      draw_styled_landmarks(image, results)

      # Show to screen
      cv2.imshow('OpenCV Feed', image)

      # Break gracefully
      if cv2.waitKey(10) & 0xFF == ord('q'):
          break
  cap.release()
  cv2.destroyAllWindows()