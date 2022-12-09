import cv2
import os
import numpy as np
root="video/subvideo"
dir="dataset"
os.makedirs(dir,exist_ok=True)
for folder in os.listdir(root):
    if "csv" not in folder:
        list_file_video=os.listdir(os.path.join(root,folder))
        for video_name in list_file_video:
            video_path=os.path.join(os.path.join(root,folder),video_name)
            video=cv2.VideoCapture(video_path)
            total_frames=int(video.get(7))
            count=0
            while True :
                ret,frame=video.read()
                count+=1
                if count ==total_frames :
                        break
                if ret and frame is not None : 
                    
                    folder_label=os.path.join(dir,folder)
                    if  not os.path.exists(folder_label):
                        os.makedirs(folder_label,exist_ok=True)
                    img_path=os.path.join(folder_label,video_name.split('.')[0]+"_"+str(count)+".jpg")
                    print(img_path)
                    cv2.imwrite(img_path,frame)