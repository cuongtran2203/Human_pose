import sys
import os
sys.path.append("..")
from preprocess import *

from config import *
VIDEO="/home/cuong/Documents/Project/Round1_mediapipe/data/video/subvideo"
if __name__ == "__main__":
    # for action in actions: 
    #     dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
    #     for sequence in range(1,no_sequences+1):
    #         try: 
    #             os.makedirs(os.path.join(DATA_PATH, action, str(dirmax+sequence)))
    #         except:
    #             pass
    for root ,dirs,list_file in os.walk(VIDEO):
        for file in list_file:

            video_path=os.path.join(root, file)
            cap = cv2.VideoCapture(video_path)
            total_frames=int(cap.get(7))
            print(total_frames)

        # Set mediapipe model 
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                
                for frame_num in range(total_frames):
                    # Read feed
                    ret, frame = cap.read()
                    if ret and frame is not None :
                        # Make detections
                        image, results = mediapipe_detection(frame, holistic)

                        # Draw landmarks
                        # draw_styled_landmarks(image, results)
                        # NEW Export keypoints
                        keypoints = extract_keypoints(results)
                        action=root.split("/")[-1]
                        video_name=file.split(".")[0]


                        npy_video_folder = os.path.join(DATA_PATH, action,video_name)
                        if not os.path.exists(npy_video_folder):
                            os.makedirs(npy_video_folder,exist_ok=True)
                        frame_path = os.path.join(npy_video_folder,str(frame_num)+".npy")
                        with open(frame_path,"wb") as f :
                            np.save(f, keypoints)
                        print("load done ",file)

                       
                        