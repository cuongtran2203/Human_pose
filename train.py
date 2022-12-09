
from model import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

label_map = {label:num for num, label in enumerate(actions)}
sq,labels=[],[]
actions=os.listdir("MP_Data")
for action in actions:
    sequences=os.listdir(os.path.join("MP_Data",action))

    for sequence in sequences:
        numpy_list_file =os.listdir(os.path.join("MP_Data",action,sequence))
        window=[]
        if len(numpy_list_file)<20:
            break
        for idx in range(20) :
            frame_num_path= os.path.join("MP_Data",action,sequence,numpy_list_file[idx])
            res=np.load(frame_num_path)
            window.append(res)
        sq.append(window)
        labels.append(label_map[action])
print(np.array(sq).shape)

X = np.array(sq)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
model=Action_Recognizer()
model.train(X_train, y_train)

            