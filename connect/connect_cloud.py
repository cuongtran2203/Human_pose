import cv2
import time
import firebase_admin
from firebase_admin import db
import threading

cred_obj = firebase_admin.credentials.Certificate('/home/cuong/Documents/GitHub/Human_pose/connect/projectt12-1fee4-firebase-adminsdk-dhgsc-0792e07a9e.json')
default_app = firebase_admin.initialize_app(cred_obj, {
	'databaseURL':"https://projectt12-1fee4-default-rtdb.asia-southeast1.firebasedatabase.app"
	})
ref_obj = db.reference("/Object Detection")
ref_human=db.reference("/Human Action")
ref_text=db.reference("/Text")
ref_request=db.reference("/Request")
def Thread_update(ref,dict_obj:dict):
    thread=threading.Thread(target=ref.set,args=(dict_obj))
    thread.start()
def read_infer_data(ref) :
    data=ref.get()
    return data
def Thread_read_data(ref,que) :
    thread=threading.Thread(target=lambda q, arg1: q.put(read_infer_data(arg1)),args=(que,ref))
    thread.start()

