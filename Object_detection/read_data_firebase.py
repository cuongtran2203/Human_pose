import cv2
import time
import firebase_admin
from firebase_admin import db
import numpy as np
import threading
from queue import Queue
cam=cv2.VideoCapture("../video.mp4")
cred_obj = firebase_admin.credentials.Certificate('/home/cuong/Documents/GitHub/Human_pose/projectt12-1fee4-firebase-adminsdk-dhgsc-57200ac9a1.json')
default_app = firebase_admin.initialize_app(cred_obj, {
	'databaseURL':"https://projectt12-1fee4-default-rtdb.asia-southeast1.firebasedatabase.app"
	})
ref = db.reference("/Object Detection")
que=Queue()
def read_infer_data(frame) :
    data=ref.get()
    return data

def Thread_update(frame):

    thread=threading.Thread(target=lambda q, arg1: q.put(read_infer_data(arg1)),args=(que,frame))
    thread.start()


if __name__ == "__main__": 

    while cam.isOpened(): 
        _,frame=cam.read()
        frame=cv2.resize(frame,(1280,720))
        t1=time.time()
        Thread_update(frame)
        if que.qsize() > 0:
            results = que.get()
            for key,value in results.items() :
                if key =="person" :
                    box=value["box"]
                    cv2.rectangle(frame, box,(0,244,24), 1)
        print("FPS : {}".format(1/(time.time()-t1)))
        cv2.imshow("sss",frame)

        if cv2.waitKey(1.5) & 0xff == ord('q') :
            break
