from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import os 
from .config import *
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
class Action_Recognizer() :
    def __init__(self) :
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(actions.shape[0], activation='softmax'))
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model.summary()
    def train(self,X_train,y_train) :
        batch_size=16
        self.model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
        earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
        mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=200, verbose=0, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)
        
