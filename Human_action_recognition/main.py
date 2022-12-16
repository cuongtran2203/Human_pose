from scipy import stats
from scipy import stats
import cv2 
import sys
# sys.path.append(".")
from preprocess import *
from config import *
from model import *
import sys
sys.path.append("..")
from Object_detection.YOLOV4.yolov4 import *
from OCR.text_recognition import *
from Object_detection.YOLOX.Infer_model_onnx_YOLOX import *
from connect.connect_cloud import *
import sys
from queue import Queue
que=Queue()
sequence = []
sentence = []
predictions = []
threshold = 0.6
colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame
model_LSTM=Action_Recognizer()
model_LSTM.model.load_weights("./model.h5")
# model_detection=Detection()
model_detection=Detection_ONNX("../Object_detection/YOLOX/yolox_nano.onnx")
model_text=Text_Recognizer()
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
        A=""
        Thread_read_data(ref_request,que)
        if que.qsize()>0:
            data=que.get()
        else :
            data=""
        # print(data) 
        draw_styled_landmarks(image, results)
        if data=="A":
            text=model_detection.detect(frame)
            object_dict={text}
            Thread_update(ref_obj,object_dict)
            print(object_dict)
        elif data=="B":
            text=model_text.text_recognizer(frame)
            dict_text={text}
            Thread_update(ref_text,dict_text)
            print(text)
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]   
        if len(sequence) == 30:
            res = model_LSTM.model.predict(np.expand_dims(sequence, axis=0))[0]
            human_dict={actions[np.argmax(res)]}
            Thread_update(ref_human,human_dict)
            predictions.append(np.argmax(res))
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
        else :
            human_dict={""}
            Thread_update(ref_human,human_dict)


            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # Show to screen
        cv2.imshow('OpenCV Feed', image)
        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
