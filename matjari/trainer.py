from matjari.load_data import get_data, load_json, load_img, load_nrows, encod, split_data
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import Sequential, layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import pandas as pd
import numpy as np
import joblib



class Trainer():
    def __init__(self, model):
        self.model = model



    def get_model(self):
        #build pipeline later
        self.model = Sequential()
        self.model.add(Rescaling(1./255, input_shape=(256, 256, 3)))

        self.model.add(layers.Conv2D(16, (10,10), activation="relu"))
        self.model.add(layers.MaxPool2D(pool_size=(3,3)))

        self.model.add(layers.Conv2D(32, (8,8), activation="relu"))
        self.model.add(layers.MaxPool2D(pool_size=(3,3)))

        self.model.add(layers.Conv2D(32, (6,6), activation="relu"))
        self.model.add(layers.MaxPool2D(pool_size=(3,3)))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1000, activation='relu'))
        self.model.add(layers.Dense(29, activation='softmax'))
        #compile
        learning_rate=1e-4
        opt = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])


    def fit(self):
        es = EarlyStopping(patience = 20, restore_best_weights = True)
        data = get_data()
        data = load_json(data)
        data = load_img(data)
        data = load_nrows(data)
        X, y = encod(data)
        X_train, X_test, y_train, y_test = split_data(X, y)
        self.model.fit(X_train, y_train,
                  batch_size=16,
                  epochs=100,
                  validation_split=0.3,
                  callbacks=[es],
                  verbose = 2)


    def evaluate(self):
        data = get_data()
        data = load_json(data)
        data = load_img(data)
        data = load_nrows(data)
        X, y = encod(data)
        X_train, X_test, y_train, y_test = split_data(X, y)
        self.model.evaluate(X_test, y_test, verbose=2)

    def save_model(self):
        joblib.dump(self.model, 'model.joblib')




if __name__ == "__main__":
    pass
    #what to test
