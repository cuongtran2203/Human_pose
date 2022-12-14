from scipy import stats
import cv2 
from preprocess import *
from config import *
from model import *
import sys
sys.path.append("..")
from connect.connect_cloud import *
colors = [(245,117,16), (117,245,16), (16,117,245),(25,69,211)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame
class Classification() :
    def __init__(self,model_path="./Human_action_recognition/model.h5") :

        # 1. New detection variables
        self.model_LSTM=Action_Recognizer()
        self.model_LSTM.model.load_weights(model_path)
    def run(self,frame):
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.7
        action=''
        # Set mediapipe model 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                frame=cv2.resize(frame,(1280,720))
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # Draw landmarks
                draw_styled_landmarks(image, results)
                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                if len(sequence) == 30:
                    res = self.model_LSTM.model.predict(np.expand_dims(sequence, axis=0))[0]
                    action=actions[np.argmax(res)]

                    dict_data={"action":action}
                    #Send data to cloud 
                    # Thread_update(ref_human,dict_data)
                    # print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                #3. Viz logic
                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]

                    # Viz probabilities
                    image = prob_viz(res, actions, image, colors)
                    
                    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, ' '.join(sentence), (3,30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    return dict_data
if __name__=="__main__":
    cam=cv2.VideoCapture(0)
    model=Classification()
    while True :
        ret,frame=cam.read()
        if ret and frame is not None :
            results=model.run(frame)
            cv2.imshow("",frame)
            if cv2.waitKey(1) & 0xff==ord("q"):
                break
        
