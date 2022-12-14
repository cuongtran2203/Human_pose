
from model import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from config import *
import numpy as np
label_map = {label:num for num, label in enumerate(actions)}
sq,labels=[],[]
actions=os.listdir(DATA_PATH)
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        print(np.array(sequences).shape)
        labels.append(label_map[action])

X = np.array(sequences)
print(X.shape)

y = to_categorical(labels).astype(np.float32)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
model=Action_Recognizer()
model.train(X_train, y_train)

            