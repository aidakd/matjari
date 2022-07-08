from matjari.load_data import get_data, get_data_using_blob, load_json, load_img, encod, split_data
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
from google.cloud import storage

BUCKET_NAME='wagon-data-839-melliani'
STORAGE_LOCATION = 'models/model_cnn.joblib'

class Trainer():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def get_model(self):
        #build pipeline later
        model = Sequential()
        model.add(Rescaling(1./255, input_shape=(256, 256, 3)))

        model.add(layers.Conv2D(16, (10,10), activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(3,3)))

        model.add(layers.Conv2D(32, (8,8), activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(3,3)))

        model.add(layers.Conv2D(32, (6,6), activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(3,3)))

        model.add(layers.Flatten())
        model.add(layers.Dense(1000, activation='relu'))
        model.add(layers.Dense(29, activation='softmax'))
        #compile
        learning_rate=1e-4
        opt = optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        print('model compiled')

    def fit(self, X, y):
        es = EarlyStopping(patience = 20, restore_best_weights = True)
        #data = get_data()
        #data = load_json(data)
        #data = load_img(data)
        #X, y = encod(data)
        #X_train, X_test, y_train, y_test = split_data(X, y)
        self.fit(X, y,
                  batch_size=16,
                  epochs=100,
                  validation_split=0.3,
                  callbacks=[es],
                  verbose = 2)
        print('model fitted')

    def evaluate(self, X, y):
        #data = get_data()
        #data = load_json(data)
        #data = load_img(data)
        #X, y = encod(data)
        #X_train, X_test, y_train, y_test = split_data(X, y)
        self.evaluate(X, y, verbose=2)
        print('evaluating done')

    def predict(self,X_new):
        y_pred = self.predict(X_new)
        return y_pred


    def upload_model_to_gcp():
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename('model.joblib')
        print("uploaded model.joblib to gcp cloud storage!")

    def save_model(self):
        joblib.dump(self, 'model.joblib')
        print('model saved')


if __name__ == "__main__":
    #what to test
    df = get_data()
    get_data_using_blob()
    df_merge = load_json(df)
    df_merge = load_img(df_merge)
    X, y = encod(df_merge)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = Trainer(X_train, y_train)
    model.fit(X_train, y_train)
